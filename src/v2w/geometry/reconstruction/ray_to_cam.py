import torch
from ..points import RayPoints, CamPoints


def reconstruct_ray_to_cam(ray_pts: RayPoints) -> CamPoints:
    """
    Reconstructs 3D points from ray to cam space.
    Args:   
        ray_pts (RayPoints): The points in the ray space.
    Returns:
        cam_pts (CamPoints): The points in the cam space.
    """
    
    # cam to ray
    # N = cam_pts.coords.shape[0]
    # r0 = cam_pts.coords[:, 0, 0] / cam_pts.coords[:, 2, 0]; 
    # r1 = cam_pts.coords[:, 1, 0] / cam_pts.coords[:, 2, 0]
    # r2 = torch.linalg.norm(cam_pts.coords.view(N, 3), dim=1)
    # ray_coords = torch.cat([r0, r1, r2], dim=1)
    
    N = ray_pts.coords.shape[0]
    cam_coords = torch.zeros((N, 3, 1), device=ray_pts.coords.device, dtype=ray_pts.coords.dtype)
    cam_coords[:, 0, 0] = ray_pts.coords[:, 0] * ray_pts.coords[:, 2]
    cam_coords[:, 1, 0] = ray_pts.coords[:, 1] * ray_pts.coords[:, 2]
    cam_coords[:, 2, 0] = ray_pts.coords[:, 2]

    J = torch.zeros((N, 3, 3), device=ray_pts.coords.device, dtype=ray_pts.coords.dtype)
    J[:, 0, 0] = 1.0 / ray_pts.coords[:, 2]
    J[:, 0, 2] = -ray_pts.coords[:, 0] / (ray_pts.coords[:, 2]**2)
    J[:, 1, 1] = 1.0 / ray_pts.coords[:, 2]
    
    J[:, 1, 2] = -cam_pts.coords[:, 1, 0] / (cam_pts.coords[:, 2, 0]**2)
    J[:, 2, 2] = 1.0 / torch.linalg.norm(cam_pts.coords.view(N, 3), dim=1)
    J_inv = torch.linalg.inv(J)
    cam_covariances = J_inv @ ray_pts.covariances @ J_inv.transpose(-2, -1)  # (N,3,3)

    cam_pts = CamPoints()
    cam_pts.coords = cam_coords
    cam_pts.covariances = cam_covariances

    return cam_pts



def reconstruct_ray_to_cam_tensor(ray_pts: RayPoints) -> CamPoints:
    """
    Reconstructs 3D points from ray to cam space.
    Args:   
        ray_pts (RayPoints): The points in the ray space.
    Returns:
        cam_pts (CamPoints): The points in the cam space.
    """
    
    # cam to ray
    # N = cam_pts.coords.shape[0]
    # r0 = cam_pts.coords[:, 0, 0] / cam_pts.coords[:, 2, 0]; 
    # r1 = cam_pts.coords[:, 1, 0] / cam_pts.coords[:, 2, 0]
    # r2 = torch.linalg.norm(cam_pts.coords.view(N, 3), dim=1)
    # ray_coords = torch.cat([r0, r1, r2], dim=1)
    
    N = ray_pts.coords.shape[0]
    cam_coords = torch.zeros((N, 3, 1), device=ray_pts.coords.device, dtype=ray_pts.coords.dtype)
    cam_coords[:, 0, 0] = ray_pts.coords[:, 0] * ray_pts.coords[:, 2]
    cam_coords[:, 1, 0] = ray_pts.coords[:, 1] * ray_pts.coords[:, 2]
    cam_coords[:, 2, 0] = ray_pts.coords[:, 2]

    J = torch.zeros((N, 3, 3), device=ray_pts.coords.device, dtype=ray_pts.coords.dtype)
    J[:, 0, 0] = 1.0 / ray_pts.coords[:, 2]
    J[:, 0, 2] = -ray_pts.coords[:, 0] / (ray_pts.coords[:, 2]**2)
    J[:, 1, 1] = 1.0 / ray_pts.coords[:, 2]
    
    J[:, 1, 2] = -cam_pts.coords[:, 1, 0] / (cam_pts.coords[:, 2, 0]**2)
    J[:, 2, 2] = 1.0 / torch.linalg.norm(cam_pts.coords.view(N, 3), dim=1)
    J_inv = torch.linalg.inv(J)
    cam_covariances = J_inv @ ray_pts.covariances @ J_inv.transpose(-2, -1)  # (N,3,3)

    cam_pts = CamPoints()
    cam_pts.coords = cam_coords
    cam_pts.covariances = cam_covariances

    return cam_pts
