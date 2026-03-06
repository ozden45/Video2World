import torch
from ..points import SFMPoints, ImagePoints
from .sfm_to_cam import project_sfm_to_cam, project_sfm_to_cam_tensor
from .cam_to_ray import project_cam_to_ray, project_cam_to_ray_tensor
from .ray_to_img import project_ray_to_img, project_ray_to_img_tensor
from ...exception import ShapeError


def project_sfm_to_img(sfm_pts: SFMPoints, W: torch.Tensor, K: torch.Tensor) -> ImagePoints:
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
    ray_pts = project_cam_to_ray(cam_pts)
    img_pts = project_ray_to_img(ray_pts, K)
    
    return img_pts


def project_sfm_to_img_tensor(sfm_coords: torch.Tensor, sfm_covariances: torch.Tensor, W: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects 3D points from world to image space.
    Args:   
        sfm_pts (SFMPoints): The points in the world space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        img_pts (ImagePoints): The points in the image space.
    """

    cam_coords, cam_covariances = project_sfm_to_cam_tensor(sfm_coords, sfm_covariances, W)
    ray_coords, ray_covariances = project_cam_to_ray_tensor(cam_coords, cam_covariances)
    img_coords, img_covariances = project_ray_to_img_tensor(ray_coords, ray_covariances, K)
    
    return img_coords, img_covariances

