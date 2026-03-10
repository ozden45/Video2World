import torch
from ..points import RayPoints, ImagePoints


def reconstruct_img_to_ray(img_pts: ImagePoints, K: torch.Tensor) -> RayPoints:
    """
    Reconstruct ray-space points from image-space points.
    """

    K = K.to(dtype=img_pts.coords.dtype, device=img_pts.coords.device)

    N = img_pts.coords.shape[0]

    # homogeneous image coords
    img_h = torch.cat(
        [img_pts.coords, torch.ones(N,1, device=img_pts.coords.device, dtype=img_pts.coords.dtype)],
        dim=1
    )

    K_inv = torch.inverse(K)

    ray_coords = (K_inv @ img_h.T).T

    J = K_inv[:, :2]  # (3,2)

    ray_covariances = (
        J.unsqueeze(0)
        @ img_pts.covariances
        @ J.transpose(0,1).unsqueeze(0)
    )

    ray_pts = RayPoints(
        coords=ray_coords,
        covariances=ray_covariances,
        colors=img_pts.colors,
        alphas=img_pts.alphas
    )

    return ray_pts


def reconstruct_img_to_ray_batched(img_pts: ImagePoints, K: torch.Tensor) -> RayPoints:

    K = K.to(dtype=img_pts.coords.dtype, device=img_pts.coords.device)

    B, N, _ = img_pts.coords.shape

    ones = torch.ones(B, N, 1, device=img_pts.coords.device, dtype=img_pts.coords.dtype)
    img_h = torch.cat([img_pts.coords, ones], dim=-1)

    K_inv = torch.inverse(K)

    ray_coords = torch.matmul(img_h, K_inv.transpose(-1,-2))

    J = K_inv[:, :, :2]

    ray_covariances = (
        J.unsqueeze(1)
        @ img_pts.covariances
        @ J.transpose(-1,-2).unsqueeze(1)
    )

    ray_pts = RayPoints(
        coords=ray_coords,
        covariances=ray_covariances,
        colors=img_pts.colors,
        alphas=img_pts.alphas
    )

    return ray_pts


def reconstruct_img_to_ray_tensor(img_pts: ImagePoints, K: torch.Tensor) -> RayPoints:
    """
    Reconstructs 3D points from image to ray space
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
