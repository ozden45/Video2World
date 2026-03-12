"""
rasterizer.py

Rasterization-based rendering of 3D points into 2D images.
"""

import torch
from typing import Tuple
from v2w.geometry.projection import *
from v2w.geometry.camera import Camera
from v2w.rendering.splat import gaussian_splat
from v2w.rendering.sh_color import sh_color
from ..geometry.points import *


def rasterize(
    sfm_coords: torch.Tensor, 
    sfm_covs: torch.Tensor, 
    sfm_colors: torch.Tensor, 
    sfm_alphas: torch.Tensor, 
    W: torch.Tensor, 
    K: torch.Tensor,
    img_size: Tuple[int, int], 
    nsigma: int = 20
    ) -> torch.Tensor:
    """
    Rasterizes 3D points into a 2D image using Gaussian splatting.
    Args:
        sfm_pts (SFMPoints): The 3D points in the world space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
        H (int): The height of the output image.
        W_img (int): The width of the output image.
    Returns:
        img (torch.Tensor): The rasterized image of shape (H, W_img, 3).
    """
    
    sfm_pts = SFMPoints(
        coords=sfm_coords,
        covariances=sfm_covs,
        colors=sfm_colors,
        alphas=sfm_alphas
    )
    
    
    # Determine the in-range points
    cam_pts = project_sfm_to_cam(sfm_pts, W)
    cam_coords, cam_covs = cam_pts.coords, cam_pts.covariances
    mask = (cam_coords[:, 0] > 0) & (cam_coords[:, 1] > 0) & (cam_coords[:, 2] > 0)
    
    # Mask the in-range points
    cam_coords, cam_covs = cam_coords[mask], cam_covs[mask]
    
    # Project the points to the image space
    img_pts = project_cam_to_img(cam_pts, K)
    img_coords, img_covs = img_pts.coords, img_pts.covariances
        
    # Calculate the view direction
    view = Camera.extrinsic_to_view(W[:3, :3])
    
    # Empty image sheet
    img = torch.zeros((img_size[0], img_size[1], 3), dtype=torch.float32, device=torch.device("cuda"))  
        
    # Project 3D points to 2D image space
    coords = img_coords
    inv_covs = torch.linalg.inv(img_covs)
    colors = sh_color(view, sfm_colors)
    alphas = sfm_alphas
    
    # Rasterize points using Gaussian splatting
    img = gaussian_splat(
        img=img,
        mu=coords,
        inv_cov=inv_covs,
        clr=colors,
        alpha=alphas,
        img_size=img_size,
        nsigma=nsigma
    )
    
    return img


