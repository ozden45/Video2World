import torch
from ..points import SFMPoints, SFMPointsBatched, CameraPoints, CameraPointsBatched
from ...exception import ShapeError



def project_sfm_to_cam(sfm_pts: SFMPoints, Rt: torch.Tensor) -> CameraPoints:
    """
    Projects SfM points from world to camera space.

    Args:
        sfm_pts (SFMPoints): The points in the world space.
        W (torch.Tensor): The extrinsic camera parameters (3x4).

    Returns:
        cam_pts (CameraPoints): The points in the camera space.
    """

    # Check the shape of W
    if Rt.shape != (3, 4):
        raise ShapeError(
            f"project_sfm_to_cam(): Invalid W shape {Rt.shape}, expected (3,4)."
        )

    # Match dtype and device
    Rt = Rt.to(dtype=sfm_pts.coords.dtype, device=sfm_pts.coords.device)

    # Extract rotation and translation
    R = Rt[:, :3]
    t = Rt[:, 3]

    cam_coords = sfm_pts.coords @ R.T + t
    cam_covariances = R @ sfm_pts.covariances @ R.T

    # Create CameraPoints
    cam_pts = CameraPoints(
        coords=cam_coords,
        covariances=cam_covariances,
        colors=sfm_pts.colors,
        alphas=sfm_pts.alphas
    )

    return cam_pts



def project_sfm_to_cam_batched(sfm_batched: SFMPointsBatched, Rt_batched: torch.Tensor) -> CameraPointsBatched:
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

    if Rt_batched.ndim != 3 or Rt_batched.shape[1:] != (3, 4):
        raise ShapeError(
            f"project_sfm_to_cam_batched(): Invalid Rt_batched shape {Rt_batched.shape}, expected (B,3,4)."
        )

    # Match dtype and device
    Rt_batched = Rt_batched.to(
        dtype=sfm_batched.coords.dtype, 
        device=sfm_batched.coords.device
    )

    R = Rt_batched[:, :, :3]
    t = Rt_batched[:, :, 3]

    # Project world points into camera space
    cam_coords = torch.einsum('bij,bnj->bni', 
                              R, 
                              sfm_batched.coords) + t.unsqueeze(1)
    cam_covariances = torch.einsum('bij,bnjk,bkl->bnil', 
                                   R, 
                                   sfm_batched.covariances, 
                                   R.transpose(-1,-2))
    
    cam_batched = CameraPointsBatched(
        coords=cam_coords,
        covariances=cam_covariances,
        colors=sfm_batched.colors,
        alphas=sfm_batched.alphas
    )

    return cam_batched

