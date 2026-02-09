# --------------------------------------------------------------
#   projection.py
#
#   Description:
#       This script includes an implementation of projection 
#       between different space coordinates
#   
#   Author: Ozden Ozel
#   Created: 2026-02-04
#
# --------------------------------------------------------------


import torch
from v2w.geometry.points import SFMPoints, CamPoints, RayPoints, ImagePoints



# --------------------------------------------------------------
# Projection functions from world to image
# --------------------------------------------------------------

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
    if W.shape == (3,4):
        raise ValueError(f"project_sfm_to_cam(): Invalid W shape {W.shape}, expected (3,4).")
    
    # Convert W to the rotation matrix (R) and translational matrix (t)
    N = sfm_pts.coords.shape[0]
    W = W.unsqueeze(0).repeat(N, 1, 1)
    R = W[:, :3, :3]
    t = W[:, :3, 3:].reshape(N, 3, 1)

    # Calculate the points of the coordinates and covariances in the camera space
    cam_coords = R @ sfm_pts.coords + t
    cam_covariances = R @ sfm_pts.covariances @ R.transpose(-2, -1)
    
    cam_pts = CamPoints()
    cam_pts.coords = cam_coords
    cam_pts.covariances = cam_covariances
    
    return cam_pts
        

def project_cam_to_ray(cam_pts: CamPoints) -> RayPoints:
    """
    Projects 3D points from camera to ray space.
    Args:   
        cam_pts (CamPoints): The points in the camera space.
    Returns:
        ray_pts (RayPoints): The points in the ray space.
    """
    N = cam_pts.coords.shape[0]
    r0 = cam_pts.coords[:, 0, 0] / cam_pts.coords[:, 2, 0]; 
    r1 = cam_pts.coords[:, 1, 0] / cam_pts.coords[:, 2, 0]
    r2 = torch.linalg.norm(cam_pts.coords.view(N, 3), dim=1)
    ray_coords = torch.cat([r0, r1, r2], dim=1)
    
    # Batched Jacobian (N,3,3)
    J = torch.zeros((N, 3, 3), device=cam_pts.coords.device, dtype=cam_pts.coords.dtype)
    J[:, 0, 0] = 1.0 / cam_pts.coords[:, 2, 0]
    J[:, 0, 2] = -cam_pts.coords[:, 0, 0] / (cam_pts.coords[:, 2, 0]**2)
    J[:, 1, 1] = 1.0 / cam_pts.coords[:, 2, 0]
    J[:, 1, 2] = -cam_pts.coords[:, 1, 0] / (cam_pts.coords[:, 2, 0]**2)
    J[:, 2, 2] = 1.0 / torch.linalg.norm(cam_pts.coords.view(N, 3), dim=1)

    ray_covariances = J @ cam_pts.covariances @ J.transpose(-2, -1)  # (N,3,3)
    
    ray_pts = RayPoints()
    ray_pts.coords = ray_coords
    ray_pts.covariances = ray_covariances
    
    return ray_pts


def project_ray_to_img(ray_pts: RayPoints, K: torch.Tensor) -> ImagePoints:
    """
    Projects 3D points from ray to image space.
    Args:   
        ray_pts (RayPoints): The points in the ray space.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        img_pts (ImagePoints): The points in the image space.
    """
    N = ray_pts.coords.shape[0]
    K = K.unsqueeze(0).repeat(N, 1, 1)
    img_coords = K @ ray_pts.coords.unsqueeze(-1)
    img_covariances = ray_pts.covariances
    
    img_pts = ImagePoints()
    img_pts.coords = img_coords
    img_pts.covariances = img_covariances
    
    return img_pts


def project_sfm_to_img(sfm_pts: SFMPoints, W: torch.Tensor, K: torch.Tensor) -> ImagePoints:
    """
    Projects 3D points from world to image space.
    Args:   
        sfm_pts (SFMPoints): The points in the world space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        img_pts (ImagePoints): The points in the image space.
    """

    cam_pts = project_sfm_to_cam(sfm_pts, W)
    ray_pts = project_cam_to_ray(cam_pts)
    img_pts = project_ray_to_img(ray_pts, K)
    
    return img_pts



# --------------------------------------------------------------
# Projection functions from image to world
# --------------------------------------------------------------

def project_img_to_ray(img_pts: ImagePoints, K: torch.Tensor) -> RayPoints:
    """
    Projects 3D points from image to ray space
    Args:   
        img_pts (ImagePoints): The points in the image space.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        ray_pts (RayPoints): The points in the ray space.
    """
    N = img_pts.coords.shape[0]
    K_inv = torch.linalg.inv(K)
    K_inv = K_inv.unsqueeze(0).repeat(N, 1, 1)
    ray_coords = K_inv @ img_pts.coords
    ray_covariances = img_pts.covariances
    
    ray_pts = RayPoints()
    ray_pts.coords = ray_coords
    ray_pts.covariances = ray_covariances
    
    return ray_pts
    

def project_ray_to_cam(ray_pts: RayPoints) -> CamPoints:
    """
    Projects 3D points from ray to cam space.
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
    cam_coords = torch.tensor
    
    J = torch.zeros((N, 3, 3), device=cam_pts.coords.device, dtype=cam_pts.coords.dtype)
    J[:, 0, 0] = 1.0 / cam_pts.coords[:, 2, 0]
    J[:, 0, 2] = -cam_pts.coords[:, 0, 0] / (cam_pts.coords[:, 2, 0]**2)
    J[:, 1, 1] = 1.0 / cam_pts.coords[:, 2, 0]
    J[:, 1, 2] = -cam_pts.coords[:, 1, 0] / (cam_pts.coords[:, 2, 0]**2)
    J[:, 2, 2] = 1.0 / torch.linalg.norm(cam_pts.coords.view(N, 3), dim=1)
    J_inv = torch.linalg.inv(J)
    cam_covariances = J_inv @ ray_pts.covariances @ J_inv.transpose(-2, -1)  # (N,3,3)

    cam_pts = CamPoints()
    cam_pts.coords = cam_coords
    cam_pts.covariances = cam_covariances

    return cam_pts


def project_cam_to_sfm(cam_pts: CamPoints, W: torch.Tensor) -> SFMPoints:
    """
    Projects 3D points from cam to world space.
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
    

def project_img_to_sfm(img_pts: ImagePoints, W: torch.Tensor, K: torch.Tensor):
    """
    Projects points from image to world space.
    Args:   
        img_pts (SFMPoints): The points in the image space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        sfm_pts (CamPoints): The points in the world space.
    """

    ray_pts = project_img_to_ray(img_pts, K)
    cam_pts = project_ray_to_cam(ray_pts)
    sfm_pts = project_cam_to_sfm(cam_pts, W)
    
    return sfm_pts


