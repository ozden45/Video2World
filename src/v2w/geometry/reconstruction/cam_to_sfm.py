import torch
from ..points import CamPoints, SFMPoints


def reconstruct_cam_to_sfm(cam_pts: CamPoints, W: torch.Tensor) -> SFMPoints:
    """
    Reconstructs 3D points from cam to world space.
    Args:   
        cam_pts (CamPoints): The points in the cam space.
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
    

def reconstruct_cam_to_sfm_tensor(cam_pts: CamPoints, W: torch.Tensor) -> SFMPoints:
    """
    Reconstructs 3D points from cam to world space.
    Args:   
        cam_pts (CamPoints): The points in the cam space.
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
    
