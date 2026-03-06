import torch
from ..points import CamPoints, RayPoints
from ...exception import ShapeError


def project_cam_to_ray(cam_pts: CamPoints) -> RayPoints:
    """
    Projects 3D points from camera to ray space.
    Args:   
        cam_pts (CamPoints): The points in the camera space.
    Returns:
        ray_pts (RayPoints): The points in the ray space.
    """
    X = cam_pts.coords 
    N = X.shape[0]

    norm = torch.linalg.norm(X, dim=1, keepdim=True)
    ray_coords = X / norm 

    J = torch.zeros((N, 3, 3), device=X.device, dtype=X.dtype)

    for i in range(3):
        for j in range(3):
            if i == j:
                J[:, i, j] = (norm.squeeze()**2 - X[:, i]**2) / norm.squeeze()**3
            else:
                J[:, i, j] = -X[:, i] * X[:, j] / norm.squeeze()**3

    ray_covariances = J @ cam_pts.covariances @ J.transpose(-2, -1)
    
    ray_pts = RayPoints(
        coords=ray_coords,
        covariances=ray_covariances,
        colors=cam_pts.colors,
        alphas=cam_pts.alphas
    )
    
    return ray_pts


def project_cam_to_ray_tensor(cam_coords: torch.Tensor, cam_covariances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects 3D points from camera to ray space.
    Args:   
        cam_pts (CamPoints): The points in the camera space.
    Returns:
        ray_pts (RayPoints): The points in the ray space.
    """
    X = cam_coords 
    N = X.shape[0]

    norm = torch.linalg.norm(X, dim=1, keepdim=True)
    ray_coords = X / norm 

    J = torch.zeros((N, 3, 3), device=X.device, dtype=X.dtype)

    for i in range(3):
        for j in range(3):
            if i == j:
                J[:, i, j] = (norm.squeeze()**2 - X[:, i]**2) / norm.squeeze()**3
            else:
                J[:, i, j] = -X[:, i] * X[:, j] / norm.squeeze()**3

    ray_covariances = J @ cam_covariances @ J.transpose(-2, -1)
    
    return ray_coords, ray_covariances