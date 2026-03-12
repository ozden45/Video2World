import torch
from ..points import CameraPoints, CameraPointsBatched, ImagePoints, ImagePointsBatched
from ...exception import ShapeError



def reconstruct_img_to_cam(
    img_pts: ImagePoints, 
    K: torch.Tensor, 
    depth: torch.Tensor
) -> CameraPoints:
    """
    """
    
    # Check the shape of K
    if K.shape != (3, 3):
        raise ShapeError(
            f"project_cam_to_img(): Invalid W shape {K.shape}, expected (3,3)."
        )
        
    # Match dtype/device
    K = K.to(dtype=img_pts.coords.dtype, device=img_pts.coords.device)
    
    # camera point
    N = depth.shape[0]
    X = depth * (img_pts.coords[:, 0] - K[0, 2]) / K[0, 0]
    Y = depth * (img_pts.coords[:, 1] - K[1, 2]) / K[1, 1]
    Z = depth
    
    J = torch.zeros((N, 3, 2))
    
    J[:, 0, 0] = depth / K[0, 0]
    J[:, 1, 1] = depth / K[1, 1]
    
    cam_coords = torch.stack([X, Y, Z], dim=1)
    cam_covariances = J @ img_pts.covariances @ J.T
    
    cam_pts = CameraPoints(
        coords=cam_coords,
        covariances=cam_covariances,
        colors=img_pts.colors,
        alphas=img_pts.alphas
    )
    
    return cam_pts
    
    

def reconstruct_img_to_cam_batched(
    img_batched: ImagePointsBatched, 
    K: torch.Tensor, 
    depth_batched: torch.Tensor
) -> CameraPointsBatched:
    """
    """
    
    # Check the shape of K
    if K.shape != (3, 3):
        raise ShapeError(
            f"project_cam_to_img(): Invalid K shape {K.shape}, expected (3,3)."
        )
        
    # Match dtype/device
    K = K.to(dtype=img_batched.coords.dtype, device=img_batched.coords.device)
    
    # camera point
    B = depth_batched.shape[0]
    N = depth_batched.shape[1]
    X = depth_batched * (img_batched.coords[:, 0] - K[0, 2]) / K[0, 0]
    Y = depth_batched * (img_batched.coords[:, 1] - K[1, 2]) / K[1, 1]
    Z = depth_batched
    
    J = torch.zeros((B, N, 3, 2))
    
    J[:, :, 0, 0] = depth_batched / K[0, 0]
    J[:, :, 1, 1] = depth_batched / K[1, 1]
    
    cam_coords = torch.stack([X, Y, Z], dim=1)
    cam_covariances = J @ img_batched.covariances @ J.T
    
    cam_batched = CameraPointsBatched(
        coords=cam_coords,
        covariances=cam_covariances,
        colors=img_batched.colors,
        alphas=img_batched.alphas
    )
    
    return cam_batched
    
    
    
    
    