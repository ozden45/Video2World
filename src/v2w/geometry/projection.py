"""
v2w.geometry.projection

This script includes an implementation of projection 
between different space coordinates.
"""

import torch
from v2w.geometry.points import SFMPoints, CamPoints, RayPoints, ImagePoints
from v2w.exception import ShapeError


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
        

def project_cam_to_ray(cam_pts: CamPoints) -> RayPoints:
    """
    Projects 3D points from camera to ray space.
    Args:   
        cam_pts (CamPoints): The points in the camera space.
    Returns:
        ray_pts (RayPoints): The points in the ray space.
    """
    X = cam_pts.coords 
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

    ray_covariances = J @ cam_pts.covariances @ J.transpose(-2, -1)
    
    ray_pts = RayPoints(
        coords=ray_coords,
        covariances=ray_covariances,
        colors=cam_pts.colors,
        alphas=cam_pts.alphas
    )
    
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
    
    # Carry W tensor to the same device and dtype as sfm_pts
    K = K.to(
        dtype=ray_pts.coords.dtype, 
        device=ray_pts.coords.device
        )
    
    N = ray_pts.coords.shape[0]
    #K = K.unsqueeze(0).repeat(N, 1, 1)
    img_coords = K @ ray_pts.coords.unsqueeze(-1)
    img_covariances = ray_pts.covariances
    
    img_pts = ImagePoints(
        coords=img_coords,
        covariances=img_covariances,
        colors=ray_pts.colors,
        alphas=ray_pts.alphas
    )
    
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


