import torch
from ..points import RayPoints, ImagePoints


def reconstruct_img_to_ray(img_pts: ImagePoints, K: torch.Tensor) -> RayPoints:
    """
    Reconstructs 3D points from image to ray space
    Args:   
        img_pts (ImagePoints): The points in the image space.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        ray_pts (RayPoints): The points in the ray space.
    """
    N = img_pts.coords.shape[0]
    K_inv = torch.linalg.inv(K)
    K_inv = K_inv.unsqueeze(0).repeat(N, 1, 1)
    ray_coords = K_inv @ img_pts.coords
    ray_covariances = img_pts.covariances
    
    ray_pts = RayPoints()
    ray_pts.coords = ray_coords
    ray_pts.covariances = ray_covariances
    
    return ray_pts


def reconstruct_img_to_ray_tensor(img_pts: ImagePoints, K: torch.Tensor) -> RayPoints:
    """
    Reconstructs 3D points from image to ray space
    Args:   
        img_pts (ImagePoints): The points in the image space.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        ray_pts (RayPoints): The points in the ray space.
    """
    N = img_pts.coords.shape[0]
    K_inv = torch.linalg.inv(K)
    K_inv = K_inv.unsqueeze(0).repeat(N, 1, 1)
    ray_coords = K_inv @ img_pts.coords
    ray_covariances = img_pts.covariances
    
    ray_pts = RayPoints()
    ray_pts.coords = ray_coords
    ray_pts.covariances = ray_covariances
    
    return ray_pts
