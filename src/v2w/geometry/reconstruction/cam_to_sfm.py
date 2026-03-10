import torch
from ..points import CameraPoints, SFMPoints


def reconstruct_cam_to_sfm(cam_pts: CameraPoints, W: torch.Tensor) -> SFMPoints:

    W = W.to(dtype=cam_pts.coords.dtype, device=cam_pts.coords.device)

    R = W[:, :3]
    t = W[:, 3]

    world_coords = (cam_pts.coords - t) @ R

    R_T = R.T

    world_covariances = (
        R_T.unsqueeze(0)
        @ cam_pts.covariances
        @ R.unsqueeze(0)
    )

    sfm_pts = SFMPoints(
        coords=world_coords,
        covariances=world_covariances,
        colors=cam_pts.colors,
        alphas=cam_pts.alphas
    )

    return sfm_pts


def reconstruct_cam_to_sfm_batched(cam_pts: CameraPoints, W: torch.Tensor):

    W = W.to(dtype=cam_pts.coords.dtype, device=cam_pts.coords.device)

    R = W[:, :, :3]
    t = W[:, :, 3]

    world_coords = torch.matmul(
        cam_pts.coords - t.unsqueeze(1),
        R
    )

    R_T = R.transpose(-1,-2)

    world_covariances = (
        R_T.unsqueeze(1)
        @ cam_pts.covariances
        @ R.unsqueeze(1)
    )

    sfm_pts = SFMPoints(
        coords=world_coords,
        covariances=world_covariances,
        colors=cam_pts.colors,
        alphas=cam_pts.alphas
    )

    return sfm_pts
    

def reconstruct_cam_to_sfm_tensor(cam_pts: CameraPoints, W: torch.Tensor) -> SFMPoints:
    """
    Reconstructs 3D points from cam to world space.
    Args:   
        cam_pts (CameraPoints): The points in the cam space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
    Returns:
        sfm_pts (SFMPoints): The points in the world space.
    """
    N = cam_pts.coords.shape[0]
    W = W.repeat(N, 1, 1)
    R = W[:, :3, :3]
    R_inv = torch.linalg.inv(R)
    t = W[:, :3, 3:].reshape(N, 3, 1)    
    
    sfm_coords = R_inv @ (cam_pts.coords - t)
    sfm_covariances = R_inv @ cam_pts.covariances @ R_inv.transpose(-2, -1)
    
    sfm_pts = SFMPoints()
    sfm_pts.coords = sfm_coords
    sfm_pts.covariances = sfm_covariances
    
    return sfm_pts
    
