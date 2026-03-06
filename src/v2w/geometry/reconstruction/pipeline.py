import torch
from ..points import ImagePoints, SFMPoints
from .img_to_ray import reconstruct_img_to_ray, reconstruct_img_to_ray_tensor
from .ray_to_cam import reconstruct_ray_to_cam, reconstruct_ray_to_cam_tensor
from .cam_to_sfm import reconstruct_cam_to_sfm, reconstruct_cam_to_sfm_tensor


def reconstruct_img_to_sfm(img_pts: ImagePoints, W: torch.Tensor, K: torch.Tensor) -> SFMPoints:
    """
    Reconstructs points from image to world space.
    Args:   
        img_pts (SFMPoints): The points in the image space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        sfm_pts (CamPoints): The points in the world space.
    """

    ray_pts = reconstruct_img_to_ray(img_pts, K)
    cam_pts = reconstruct_ray_to_cam(ray_pts)
    sfm_pts = reconstruct_cam_to_sfm(cam_pts, W)
    
    return sfm_pts


def reconstruct_img_to_sfm_tensor(img_pts: ImagePoints, W: torch.Tensor, K: torch.Tensor):
    """
    Reconstructs points from image to world space.
    Args:   
        img_pts (SFMPoints): The points in the image space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        sfm_pts (CamPoints): The points in the world space.
    """

    ray_pts = reconstruct_img_to_ray_tensor(img_pts, K)
    cam_pts = reconstruct_ray_to_cam_tensor(ray_pts)
    sfm_pts = reconstruct_cam_to_sfm_tensor(cam_pts, W)
    
    return sfm_pts