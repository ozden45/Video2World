import torch
from ..points import CameraPoints, CameraPointsBatched, SFMPoints, SFMPointsBatched
from ...exception import ShapeError



def reconstruct_cam_to_sfm(
    cam_pts: CameraPoints, 
    W: torch.Tensor
) -> SFMPoints:
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

    W = W.to(
        dtype=cam_pts.coords.dtype, 
        device=cam_pts.coords.device
    )

    R = W[:, :3]
    t = W[:, 3]

    sfm_coords = (cam_pts.coords - t) @ R
    sfm_covariances = R.T @ cam_pts.covariances @ R

    sfm_pts = SFMPoints(
        coords=sfm_coords,
        covariances=sfm_covariances,
        colors=cam_pts.colors,
        alphas=cam_pts.alphas
    )

    return sfm_pts


def reconstruct_cam_to_sfm_batched(
    cam_batched: CameraPointsBatched, 
    W_batched: torch.Tensor
) -> SFMPointsBatched:
    """
    """
    
    # Check the shape of W
    if W_batched.ndim != 3 or W_batched.shape[1:] != (3, 4):
        raise ShapeError(
            f"project_sfm_to_cam(): Invalid W shape {W.shape}, expected (3,4)."
        )

    W_batched = W_batched.to(
        dtype=cam_batched.coords.dtype, 
        device=cam_batched.coords.device
    )

    R = W_batched[:, :, :3]
    t = W_batched[:, :, 3]

    sfm_coords = torch.einsum('bij,bnj->bni',
                              R.transpose(-1,-2),
                              cam_batched.coords) - t.unsqueeze(1)
    sfm_covariances = torch.einsum('bij,bnjk,bkl->bnil',
                                   R.transpose(-1,-2),
                                   cam_batched.covariances,
                                   R)

    sfm_batched = SFMPointsBatched(
        coords=sfm_coords,
        covariances=sfm_covariances,
        colors=cam_batched.colors,
        alphas=cam_batched.alphas
    )

    return sfm_batched
    
    