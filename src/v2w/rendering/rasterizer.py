"""
rasterizer.py

Rasterization-based rendering of 3D points into 2D images.
"""

import torch
from typing import Tuple
from v2w.geometry.projection import project_sfm_to_cam_tensor, project_cam_to_ray_tensor, project_ray_to_img_tensor
from v2w.geometry.camera import extrinsic_to_view
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
    
    # Determine the in-range points
    cam_coords, cam_covs = project_sfm_to_cam_tensor(sfm_coords, sfm_covs, W)
    mask = (cam_coords[:, 0] > 0) & (cam_coords[:, 1] > 0) & (cam_coords[:, 2] > 0)
    bounds = mask.nonzero(as_tuple=True)
    
    # Determine the bounding box of the in-range points
    x_min, y_min, z_min = bounds.min(dim=0).values
    x_max, y_max, z_max = bounds.max(dim=0).values
    
    # Project the points to the image space
    ray_coords, ray_covariances = project_cam_to_ray_tensor(
        cam_coords[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],
        cam_covs[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],
    )
    
    img_coords, img_covs = project_ray_to_img_tensor(
        ray_coords[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],
        ray_covariances[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],
        K
    )
        
    # Calculate the view direction
    view = extrinsic_to_view(W[:3, :3])
    
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


