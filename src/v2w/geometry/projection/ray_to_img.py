import torch
from typing import Tuple
from ..points import RayPoints, ImagePoints
from ...exception import ShapeError



def project_ray_to_img(ray_pts: RayPoints, K: torch.Tensor) -> ImagePoints:
    """
    Projects 3D points from ray space to image space.

    Args:
        ray_pts (RayPoints): Points in ray space (N,3)
        K (torch.Tensor): Intrinsic matrix (3,3)

    Returns:
        ImagePoints: Points in image space
    """

    K = K.to(
        dtype=ray_pts.coords.dtype,
        device=ray_pts.coords.device
    )

    X = ray_pts.coords                    # (N,3)

    # ---- projection ----
    img_h = (K @ X.T).T                   # (N,3)
    img_coords = img_h[:, :2]             # (N,2)

    # ---- covariance propagation ----
    J = K[:2, :]                          # (2,3)

    img_covariances = (
        J.unsqueeze(0)
        @ ray_pts.covariances
        @ J.transpose(0,1).unsqueeze(0)
    )                                     # (N,2,2)

    img_pts = ImagePoints(
        coords=img_coords,
        covariances=img_covariances,
        colors=ray_pts.colors,
        alphas=ray_pts.alphas
    )

    return img_pts





def project_ray_to_img_batched(ray_pts: RayPoints, K: torch.Tensor) -> ImagePoints:
    """
    Batched projection from ray space to image space.

    Args:
        ray_pts.coords: (B,N,3)
        ray_pts.covariances: (B,N,3,3)
        K: (B,3,3)

    Returns:
        ImagePoints
            coords: (B,N,2)
            covariances: (B,N,2,2)
    """

    K = K.to(
        dtype=ray_pts.coords.dtype,
        device=ray_pts.coords.device
    )

    X = ray_pts.coords                # (B,N,3)

    # ---- projection ----
    img_h = torch.matmul(
        X,
        K.transpose(-1, -2)
    )                                 # (B,N,3)

    img_coords = img_h[..., :2]       # (B,N,2)

    # ---- covariance propagation ----
    J = K[:, :2, :]                   # (B,2,3)

    img_covariances = (
        J.unsqueeze(1)
        @ ray_pts.covariances
        @ J.transpose(-1, -2).unsqueeze(1)
    )                                 # (B,N,2,2)

    img_pts = ImagePoints(
        coords=img_coords,
        covariances=img_covariances,
        colors=ray_pts.colors,
        alphas=ray_pts.alphas
    )

    return img_pts




def project_ray_to_img_tensor(ray_coords: torch.Tensor, ray_covariances: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        dtype=ray_coords.dtype, 
        device=ray_coords.device
        )
    
    N = ray_coords.shape[0]
    #K = K.unsqueeze(0).repeat(N, 1, 1)
    img_coords = (K @ ray_coords.T).T
    img_covariances = ray_covariances
    
    return img_coords[:, :2], img_covariances[:, :2, :2]
