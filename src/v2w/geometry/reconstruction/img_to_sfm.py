import torch
from ..points import ImagePoints, ImagePointsBatched, SFMPoints, SFMPointsBatched
from .img_to_cam import *
from .cam_to_sfm import *


def reconstruct_img_to_sfm(
    img_pts: ImagePoints, 
    W: torch.Tensor, 
    K: torch.Tensor, 
    depth: torch.Tensor
) -> SFMPoints:
    """
    Reconstructs points from image to world space.
    Args:   
        img_pts (SFMPoints): The points in the image space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        sfm_pts (CameraPoints): The points in the world space.
    """

    cam_pts = reconstruct_img_to_cam(img_pts, K, depth)
    sfm_pts = reconstruct_cam_to_sfm(cam_pts, W)
    
    return sfm_pts


def reconstruct_img_to_sfm_batched(
    img_b: ImagePointsBatched, 
    W_b: torch.Tensor, 
    K: torch.Tensor,
    depth: torch.Tensor
) -> SFMPoints:
    """
    Reconstructs points from image to world space.
    Args:   
        img_pts (SFMPoints): The points in the image space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        sfm_pts (CameraPoints): The points in the world space.
    """

    cam_b = reconstruct_img_to_cam_batched(img_b, K, depth)
    sfm_b = reconstruct_cam_to_sfm_batched(cam_b, W_b)
    
    return sfm_b
