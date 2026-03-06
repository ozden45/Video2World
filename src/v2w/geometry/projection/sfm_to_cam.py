import torch
from ..points import SFMPoints, CamPoints
from ...exception import ShapeError


def project_sfm_to_cam(sfm_pts: SFMPoints, W: torch.Tensor) -> CamPoints:
    """
    Projects SfM points from world to camera space.
    Args:   
        sfm_pts (SFMPoints): The points in the world space.
        W (torch.Tensor): The extrinsic camera parameters.
    Returns:
        cam_pts (CamPoints): The points in the camera space.
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
    cam_coords = (R @ sfm_pts.coords.T).T + t
    cam_covariances = R @ sfm_pts.covariances @ R.transpose(-2, -1)
    
    # Create the CamPoints object
    cam_pts = CamPoints(
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
        cam_pts (CamPoints): The points in the camera space.
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