import torch
from typing import Tuple
from ..points import CameraPoints, RayPoints
from ...exception import ShapeError


def project_cam_to_ray(cam_pts: CameraPoints) -> RayPoints:
    """
    Projects 3D points from camera space to ray space.

    Args:
        cam_pts (CameraPoints): Points in camera space (N,3)

    Returns:
        RayPoints: Points in ray space
    """

    X = cam_pts.coords                     # (N,3)
    N = X.shape[0]

    norm = torch.linalg.norm(X, dim=1, keepdim=True).clamp(min=1e-9)  # (N,1)

    ray_coords = X / norm                  # (N,3)

    # ---- Jacobian of normalization ----
    I = torch.eye(3, device=X.device, dtype=X.dtype).expand(N, 3, 3)

    xxT = X.unsqueeze(-1) @ X.unsqueeze(-2)   # (N,3,3)

    J = (I - xxT / norm.pow(2).unsqueeze(-1)) / norm.unsqueeze(-1)

    # ---- Covariance propagation ----
    ray_covariances = J @ cam_pts.covariances @ J.transpose(-2, -1)

    ray_pts = RayPoints(
        coords=ray_coords,
        covariances=ray_covariances,
        colors=cam_pts.colors,
        alphas=cam_pts.alphas
    )

    return ray_pts




def project_cam_to_ray_batched(cam_pts: CameraPoints) -> RayPoints:
    """
    Batched projection from camera space to ray space.

    Args:
        cam_pts.coords: (B,N,3)
        cam_pts.covariances: (B,N,3,3)

    Returns:
        RayPoints
            coords: (B,N,3)
            covariances: (B,N,3,3)
    """

    X = cam_pts.coords                       # (B,N,3)

    norm = torch.linalg.norm(X, dim=-1, keepdim=True).clamp(min=1e-9)

    ray_coords = X / norm                    # (B,N,3)

    B, N, _ = X.shape

    I = torch.eye(3, device=X.device, dtype=X.dtype).view(1,1,3,3)

    xxT = X.unsqueeze(-1) @ X.unsqueeze(-2)  # (B,N,3,3)

    J = (I - xxT / norm.pow(2).unsqueeze(-1)) / norm.unsqueeze(-1)

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
        cam_pts (CameraPoints): The points in the camera space.
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