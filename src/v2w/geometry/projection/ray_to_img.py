import torch
from ..points import RayPoints, ImagePoints
from ...exception import ShapeError



def project_ray_to_img(ray_pts: RayPoints, K: torch.Tensor) -> ImagePoints:
    """
    Projects 3D points from ray to image space.
    Args:   
        ray_pts (RayPoints): The points in the ray space.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        img_pts (ImagePoints): The points in the image space.
    """
    
    # Carry W tensor to the same device and dtype as sfm_pts
    K = K.to(
        dtype=ray_pts.coords.dtype, 
        device=ray_pts.coords.device
        )
    
    N = ray_pts.coords.shape[0]
    #K = K.unsqueeze(0).repeat(N, 1, 1)
    img_coords = (K @ ray_pts.coords.T).T
    img_covariances = ray_pts.covariances
    
    img_pts = ImagePoints(
        coords=img_coords[:, :2],
        covariances=img_covariances[:, :2, :2],
        colors=ray_pts.colors,
        alphas=ray_pts.alphas
    )
    
    return img_pts


def project_ray_to_img_tensor(ray_coords: torch.Tensor, ray_covariances: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects 3D points from ray to image space.
    Args:   
        ray_pts (RayPoints): The points in the ray space.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        img_pts (ImagePoints): The points in the image space.
    """
    
    # Carry W tensor to the same device and dtype as sfm_pts
    K = K.to(
        dtype=ray_coords.dtype, 
        device=ray_coords.device
        )
    
    N = ray_coords.shape[0]
    #K = K.unsqueeze(0).repeat(N, 1, 1)
    img_coords = (K @ ray_coords.T).T
    img_covariances = ray_covariances
    
    return img_coords[:, :2], img_covariances[:, :2, :2]
