import torch
from typing import Tuple
from ..points import SFMPoints, CameraPoints
from ...exception import ShapeError




def project_sfm_to_cam(sfm_pts: SFMPoints, W: torch.Tensor) -> CameraPoints:
    """
    Projects SfM points from world to camera space.

    Args:
        sfm_pts (SFMPoints): The points in the world space.
        W (torch.Tensor): The extrinsic camera parameters (3x4).

    Returns:
        cam_pts (CameraPoints): The points in the camera space.
    """

    # Check the shape of W
    if W.shape != (3, 4):
        raise ShapeError(
            f"project_sfm_to_cam(): Invalid W shape {W.shape}, expected (3,4)."
        )

    # Match dtype/device
    W = W.to(dtype=sfm_pts.coords.dtype, device=sfm_pts.coords.device)

    # Extract rotation and translation
    R = W[:, :3]                 # (3,3)
    t = W[:, 3]                  # (3,)

    # Transform coordinates
    cam_coords = sfm_pts.coords @ R.T + t

    # Transform covariances
    cam_covariances = R @ sfm_pts.covariances @ R.T

    # Create CameraPoints
    cam_pts = CameraPoints(
        coords=cam_coords,
        covariances=cam_covariances,
        colors=sfm_pts.colors,
        alphas=sfm_pts.alphas
    )

    return cam_pts



def project_sfm_to_cam_batched(sfm_pts: SFMPoints, W: torch.Tensor) -> CameraPoints:
    """
    Projects SfM points from world to camera space for multiple cameras.

    Args:
        sfm_pts (SFMPoints): World-space points (N,3)
        W (torch.Tensor): Batched extrinsic matrices (B,3,4)

    Returns:
        CameraPoints:
            coords: (B,N,3)
            covariances: (B,N,3,3)
    """

    if W.ndim != 3 or W.shape[1:] != (3, 4):
        raise ShapeError(
            f"project_sfm_to_cam_batched(): Invalid W shape {W.shape}, expected (B,3,4)."
        )

    W = W.to(dtype=sfm_pts.coords.dtype, device=sfm_pts.coords.device)

    B = W.shape[0]
    N = sfm_pts.coords.shape[0]

    R = W[:, :, :3]  # (B,3,3)
    t = W[:, :, 3]   # (B,3)

    # ---- Coordinates ----
    # (B,N,3) = (B,3,3) x (N,3)
    cam_coords = torch.matmul(
        sfm_pts.coords.unsqueeze(0),   # (1,N,3)
        R.transpose(1,2)               # (B,3,3)
    ) + t.unsqueeze(1)                 # (B,N,3)

    # ---- Covariances ----
    cov = sfm_pts.covariances.unsqueeze(0)  # (1,N,3,3)

    cam_covariances = torch.matmul(
        torch.matmul(R.unsqueeze(1), cov),  # (B,N,3,3)
        R.transpose(1,2).unsqueeze(1)       # (B,N,3,3)
    )

    cam_pts = CameraPoints(
        coords=cam_coords,
        covariances=cam_covariances,
        colors=sfm_pts.colors,
        alphas=sfm_pts.alphas
    )

    return cam_pts






def project_sfm_to_cam_tensor(sfm_coords: torch.Tensor, sfm_covariances: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects SfM points from world to camera space.
    Args:   
        sfm_pts (SFMPoints): The points in the world space.
        W (torch.Tensor): The extrinsic camera parameters.
    Returns:
        cam_pts (CameraPoints): The points in the camera space.
    """
    # Check the shape of W
    if W.shape != (3, 4):
        raise ShapeError(f"project_sfm_to_cam(): Invalid W shape {W.shape}, expected (3,4).")
    
    # Carry W tensor to the same device and dtype as sfm_pts
    W = W.to(
        dtype=sfm_pts.coords.dtype, 
        device=sfm_pts.coords.device
        )
    
    # Convert W to the rotation matrix (R) and translational matrix (t)
    R = W[:3, :3]
    t = W[:3, 3:].reshape(3, 1)

    # Calculate the points of the coordinates and covariances in the camera space
    cam_coords = (R @ sfm_coords.T).T + t
    cam_covariances = R @ sfm_covariances @ R.transpose(-2, -1)
    
    return cam_coords, cam_covariances

