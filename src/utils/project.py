# --------------------------------------------------------------
#   project.py
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
from .points import SFMPoints, CamPoints, RayPoints, ImagePoints


# --------------------------------------------------------------
# Projection functions from image to world
# --------------------------------------------------------------

def project_image_to_ray(image_pts: ImagePoints, K: torch.Tensor) -> RayPoints:
    """
    Projects 3D points from world to image
    Args:   
        sfm_pts (SFMPoints): -----------------
        W (torch.Tensor): -----------------
    Returns:
        cam_pts (CamPoints): -----------------
    """
    N = image_pts.coords.shape[0]
    K_inv = torch.linalg.inv(K)
    K_inv = K_inv.unsqueeze(0).repeat(N, 1, 1)
    ray_coords = K_inv @ image_pts.coords
    ray_covariances = image_pts.covariances
    
    ray_pts = RayPoints()
    ray_pts.coords = ray_coords
    ray_pts.covariances = ray_covariances
    
    return ray_pts
    

def project_ray_to_cam(ray_pts: RayPoints) -> CamPoints:
    """
    Projects 3D points from world to image
    Args:   
        sfm_pts (SFMPoints): -----------------
        W (torch.Tensor): -----------------
    Returns:
        cam_pts (CamPoints): -----------------
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
    Projects 3D points from world to image
    Args:   
        sfm_pts (SFMPoints): -----------------
        W (torch.Tensor): -----------------
    Returns:
        cam_pts (CamPoints): -----------------
    """
    N = cam_pts.coords.shape[0]
    W = W.unsqueeze(0).repeat(N, 1, 1)
    R = W[:, :3, :3]
    R_inv = torch.linalg.inv(R)
    t = W[:, :3, 3:].reshape(N, 3, 1)    
    
    sfm_coords = R_inv @ (cam_pts.coords - t)
    sfm_covariances = R_inv @ cam_pts.covariances @ R_inv.transpose(-2, -1)
    
    sfm_pts = SFMPoints()
    sfm_pts.coords = sfm_coords
    sfm_pts.covariances = sfm_covariances
    
    return sfm_pts
    

def project_image_to_sfm(image_pts: ImagePoints, W: torch.Tensor, K: torch.Tensor):
    """
    Projects 3D points from world to image
    Args:   
        sfm_pts (SFMPoints): -----------------
        W (torch.Tensor): -----------------
    Returns:
        cam_pts (CamPoints): -----------------
    """

    ray_pts = project_image_to_ray(image_pts, K)
    cam_pts = project_ray_to_cam(ray_pts)
    sfm_pts = project_cam_to_sfm(cam_pts)
    
    return sfm_pts


# --------------------------------------------------------------
# Projection functions from world to image
# --------------------------------------------------------------

def project_sfm_to_cam(sfm_pts: SFMPoints, W: torch.Tensor) -> CamPoints:
    """
    Projects 3D points from world to image
    Args:   
        sfm_pts (SFMPoints): -----------------
        W (torch.Tensor): -----------------
    Returns:
        cam_pts (CamPoints): -----------------
    """
    N = sfm_pts.coords.shape[0]
    W = W.unsqueeze(0).repeat(N, 1, 1)
    R = W[:, :3, :3]
    t = W[:, :3, 3:].reshape(N, 3, 1)

    pts = sfm_pts.coords.unsqueeze(-1)
    cam_coords = R @ pts + t
    cam_covariances = R @ sfm_pts.covariances @ R.transpose(-2, -1)
    
    cam_pts = CamPoints()
    cam_pts.coords = cam_coords
    cam_pts.covariances = cam_covariances
    
    return cam_pts
    

def project_cam_to_ray(cam_pts: CamPoints) -> RayPoints:
    """
    Projects 3D points from world to image
    Args:   
        cam_pts (CamPoints): -----------------
    Returns:
        ray_pts (RayPoints): -----------------
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
    Projects 3D points from world to image
    Args:   
        points (torch.Tensor): -----------------
        K (torch.Tensor): -----------------
    Returns:
        img_points, img_covariances (Tuple[torch.Tensor, torch.Tensor]): -----------------
    """
    N = ray_pts.coords.shape[0]
    K = K.unsqueeze(0).repeat(N, 1, 1)
    img_coords = (K @ ray_pts.coords.unsqueeze(-1)).squeeze(-1)
    img_covariances = ray_pts.covariances
    
    img_pts = ImagePoints()
    img_pts.coords = img_coords
    img_pts.covariances = img_covariances


def project_sfm_to_img(sfm_pts: SFMPoints, W: torch.Tensor, K: torch.Tensor) -> ImagePoints:
    """
    Projects 3D points from world to image
    Args:   
        sfm_pts (SFMPoints): -----------------
        W (torch.Tensor): -----------------
        K (torch.Tensor): -----------------
    Returns:
        img_pts (ImagePoints): -----------------
    """

    cam_pts = project_sfm_to_cam(sfm_pts, W)
    ray_pts = project_cam_to_ray(cam_pts)
    img_pts = project_ray_to_img(ray_pts, K)
    
    return img_pts


