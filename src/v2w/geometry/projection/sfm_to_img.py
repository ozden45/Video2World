import torch
from ..points import SFMPoints, SFMPointsBatched, ImagePoints, ImagePointsBatched
from .sfm_to_cam import *
from .cam_to_img import *
from ...exception import ShapeError



def project_sfm_to_img(
    sfm_pts: SFMPoints, 
    W: torch.Tensor,
    K: torch.Tensor
) -> ImagePoints:
    """
    Projects 3D points from world to image space.
    Args:   
        sfm_pts (SFMPoints): The points in the world space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        img_pts (ImagePoints): The points in the image space.
    """

    cam_pts = project_sfm_to_cam(sfm_pts, W)
    img_pts = project_cam_to_img(cam_pts, K)
    
    return img_pts


def project_sfm_to_img_batched(
    sfm_batched: SFMPointsBatched, 
    W_batched: torch.Tensor, 
    K: torch.Tensor
) -> ImagePointsBatched:
    """
    Projects 3D points from world to image space.
    Args:   
        sfm_pts (SFMPoints): The points in the world space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        img_pts (ImagePoints): The points in the image space.
    """
    
    cam_batched = project_sfm_to_cam_batched(sfm_batched, W_batched)
    img_batched = project_cam_to_img_batched(cam_batched, K)
    
    return img_batched
    
