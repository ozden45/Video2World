"""
rasterizer.py

Rasterization-based rendering of 3D points into 2D images.
"""

import torch
from v2w.geometry.projection import project_sfm_to_img
from v2w.geometry.points import SFMPoints, ImagePoints
from v2w.rendering.splat import gaussian_splat, gaussian_splat_ext


def rasterize_points(sfm_pts: SFMPoints, W: torch.Tensor, K: torch.Tensor, H: int, W_img: int) -> torch.Tensor:
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
    
    # Project 3D points to 2D image space
    img_pts = project_sfm_to_img(sfm_pts, W, K)
    
    # Rasterize points using Gaussian splatting
    img = gaussian_splat_ext(
        mu=img_pts.coords,
        inv_cov=img_pts.covariances,
        clr=img_pts.colors,
        alpha=img_pts.alphas,
        H=H,
        W=W_img
    )
    
    return img


