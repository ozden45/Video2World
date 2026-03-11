import torch
from ..points import CameraPoints, CameraPointsBatched, ImagePoints, ImagePointsBatched
from ...exception import ShapeError



def project_cam_to_img(cam_pts: CameraPoints, K: torch.Tensor) -> ImagePoints:
    """
    """
    
    # Check the shape of K
    if K.shape != (3, 3):
        raise ShapeError(
            f"project_cam_to_img(): Invalid W shape {K.shape}, expected (3,3)."
        )

    # Match dtype/device
    K = K.to(dtype=cam_pts.coords.dtype, device=cam_pts.coords.device)
    
    # Compute jacobian matrix
    N = cam_pts.coords.shape[0]
    X = cam_pts.coords[:, 0]
    Y = cam_pts.coords[:, 1]
    Z = cam_pts.coords[:, 2]
    
    J = torch.tensor((N, 2, 3))
    
    J[:, 0, 0] = K[0, 0] / Z
    J[:, 0, 2] = -K[0, 0] * X / (Z**2)
    J[:, 1, 1] = K[1, 1] / Z
    J[:, 1, 2] = -K[1, 1] * Y / (Z**2)
    
    # Project camera points into image space
    img_coords = cam_pts.coords @ K.T
    img_covariances = J @ cam_pts.covariances @ J.T
    
    img_pts = ImagePoints(
        coords=img_coords,
        covariances=img_covariances,
        colors=cam_pts.colors,
        alphas=cam_pts.alphas
    )

    return img_pts
    


def project_cam_to_img_batched(cam_pts: CameraPointsBatched, K: torch.Tensor) -> ImagePointsBatched:
    """
    """
    
    # Check the shape of K
    if K.shape != (3, 3):
        raise ShapeError(
            f"project_cam_to_img(): Invalid K shape {K.shape}, expected (3, 3)."
        )
    
    # Match dtype and device
    K = K.to(dtype=cam_pts.coords.dtype, device=cam_pts.coords.device)
    
    # Compute jacobian matrix
    B = cam_pts.coords.shape[0]
    N = cam_pts.coords.shape[1]
    X = cam_pts.coords[:, :, 0]
    Y = cam_pts.coords[:, :, 1]
    Z = cam_pts.coords[:, :, 2]
    
    J = torch.tensor((B, N, 2, 3))
    
    J[:, :, 0, 0] = K[0, 0] / Z
    J[:, :, 0, 2] = -K[0, 0] * X / (Z**2)
    J[:, :, 1, 1] = K[1, 1] / Z
    J[:, :, 1, 2] = -K[1, 1] * Y / (Z**2)
    
    # Project camera points into image space
    img_coords = torch.einsum('bij,bnj->bni', 
                              K, 
                              cam_pts.coords)
    img_covariances = torch.einsum('bij,bnjk,bkl->bnil', 
                                   J, 
                                   cam_pts.covariances, 
                                   J.transpose(-1,-2))
    
    img_covariances = J @ cam_pts.covariances @ J.T
    
    img_batched = ImagePointsBatched(
        coords=img_coords,
        covariances=img_covariances,
        colors=cam_pts.colors,
        alphas=cam_pts.alphas
    )
    

    return img_batched
    
    
    