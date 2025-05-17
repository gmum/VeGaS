#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from models.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
from utils.sh_utils import eval_sh
import trimesh
from typing import Union
import os
import copy


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def norm_gauss(m, sigma, t):
    log = ((m - t)**2 / sigma**2) / -2
    return torch.exp(log)


def points_gaussians(pc: PointsGaussianModel, model_path, obj_path, viewpoint_camera, pipe, modify_mesh=None):
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    mesh_scene = trimesh.load(obj_path, force="mesh")
    triangles = torch.tensor(mesh_scene.triangles).cuda().float()

    pc.load_ply(model_path)
    opacity = pc.get_opacity
    means3D = pc.get_xyz

    if modify_mesh:
        time = viewpoint_camera.time / torch.numel(pc.time_func)
        triangles = modify_mesh(triangles, time)

    pc.triangles = triangles
    _xyz = triangles[:, 0]

    means3D = _xyz
    means2D = screenspace_points

    pc.prepare_scaling_rot()
    means3D[:, 1] -= 0.001

    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs_view = pc.get_features.transpose(
        1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = (
        _xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return {"means3D": means3D,
            "means2D": means2D,
            "colors_precomp": colors_precomp,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations}


def slice_gaussians(pc: GaussianModel, viewpoint_camera, pipe, foreground=False):
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    _xyz = pc.get_xyz
    means3D = _xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    time_func = pc.get_time
    time = viewpoint_camera.time
    time = 0 + torch.sum(time_func[:time]).repeat(means3D.shape[0], 1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        rotations = pc.get_rotation

    # shape: [num_gaussians, 2 * polynomial_degree] -> [num_gaussians, 2] x polynomial_degree
    poly_weights = torch.chunk(pc._w1, chunks=pc.polynomial_degree, dim=-1)

    means3D = means3D[:, [0, -1]]
    center_gaussians = pc.get_m - time[0]
    for i, poly_weight in enumerate(poly_weights):
        means3D = means3D + poly_weight * (center_gaussians ** (i+1))

    if foreground:
        means3D = torch.cat([means3D[:, 0].unsqueeze(1),
                            torch.zeros(means3D[:, 0].shape).unsqueeze(
                                1).cuda() - 0.001,
                            means3D[:, -1].unsqueeze(1)], dim=1)
    else:
        means3D = torch.cat([means3D[:, 0].unsqueeze(1),
                            torch.zeros(means3D[:, 0].shape).unsqueeze(
                                1).cuda(),
                            means3D[:, -1].unsqueeze(1)], dim=1)

    delta = norm_gauss(pc.get_m.squeeze(),
                       pc.get_sigma.squeeze(), time[0]).unsqueeze(-1)
    scales = delta * pc.get_scaling

    mask1 = (delta > 0.01).all(dim=1)
    s = scales[:, [0, -1]]
    mask2 = (s > 0.0001).all(dim=1)
    mask = torch.logical_and(mask1, mask2)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    shs_view = pc.get_features.transpose(
        1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (
        pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return {"means3D": means3D[mask],
            "means2D": means2D[mask],
            "colors_precomp": colors_precomp[mask],
            "opacity": opacity[mask],
            "scales": scales[mask],
            "rotations": rotations[mask]}


def render(viewpoint_camera, pc_fg: Union[GaussianModel, PointsGaussianModel], pc_bg: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, modify_mesh=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    viewpoint_camera.camera_center = viewpoint_camera.camera_center
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc_fg.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if pc_bg != None:
        background_viewpoint_camera = copy.deepcopy(viewpoint_camera)
        if pipe.bg_stop_frame == -1 or background_viewpoint_camera.time < pipe.bg_stop_frame:
            bg_dict = slice_gaussians(
                pc_bg, background_viewpoint_camera, pipe, False)
        else:
            background_viewpoint_camera.time = pipe.bg_stop_frame
            bg_dict = slice_gaussians(
                pc_bg, background_viewpoint_camera, pipe, False)
    else:
        bg_dict = None

    if isinstance(pc_fg, PointsGaussianModel):
        fg_dict = points_gaussians(pc_fg, os.path.join(pipe.obj_path, f"{viewpoint_camera.time}.ply"), os.path.join(
            pipe.obj_path, f"{viewpoint_camera.time}.obj"), viewpoint_camera, pipe, modify_mesh=modify_mesh)
    else:
        fg_dict = slice_gaussians(pc_fg, viewpoint_camera, pipe, True)

    if bg_dict:
        means3D = torch.cat([bg_dict['means3D'], fg_dict['means3D']], dim=0)
        means2D = torch.cat([bg_dict['means2D'], fg_dict['means2D']], dim=0)
        colors_precomp = torch.cat(
            [bg_dict['colors_precomp'], fg_dict['colors_precomp']], dim=0)
        opacity = torch.cat([bg_dict['opacity'], fg_dict['opacity']], dim=0)
        rotations = torch.cat(
            [bg_dict['rotations'], fg_dict['rotations']], dim=0)
        scales = torch.cat([bg_dict['scales'], fg_dict['scales']], dim=0)
    else:
        means3D = fg_dict['means3D']
        means2D = fg_dict['means2D']
        colors_precomp = fg_dict['colors_precomp']
        opacity = fg_dict['opacity']
        rotations = fg_dict['rotations']
        scales = fg_dict['scales']

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image}
