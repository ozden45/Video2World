"""
rasterizer.py

Rasterization-based rendering of 3D points into 2D images.
"""

import torch
from typing import Tuple
from v2w.geometry.projection import project_sfm_to_img_tensor
from v2w.rendering.splat import gaussian_splat
from v2w.rendering.sh_color import sh_color

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
    
    # Empty image sheet
    H, W = img_size[0], img_size[1]
    img = torch.zeros((H, W, 3), dtype=torch.float32, device=torch.device("cuda"))  
    
    # Project 3D points to 2D image space
    coords, covs = project_sfm_to_img_tensor(sfm_coords, sfm_covs, W, K)
    inv_covs = torch.linalg.inv(covs)
    colors = sfm_colors
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


