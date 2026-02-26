"""
v2w.geometry.reconstruction

Functions for reconstructing image pixel to 
3D points.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Iterable, Tuple
from v2w.geometry.points import SFMPoints, CamPoints, RayPoints, ImagePoints, SFMPointCloud
from v2w.io import load_intrinsic_mat
from v2w.utils.misc import is_path_exists


def reconstruct_img_to_ray(img_pts: ImagePoints, K: torch.Tensor) -> RayPoints:
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
    

def reconstruct_img_to_sfm(img_pts: ImagePoints, W: torch.Tensor, K: torch.Tensor):
    """
    Reconstructs points from image to world space.
    Args:   
        img_pts (SFMPoints): The points in the image space.
        W (torch.Tensor): The extrinsic camera parameter matrix.
        K (torch.Tensor): The intrinsic camera parameter matrix.
    Returns:
        sfm_pts (CamPoints): The points in the world space.
    """

    ray_pts = reconstruct_img_to_ray(img_pts, K)
    cam_pts = reconstruct_ray_to_cam(ray_pts)
    sfm_pts = reconstruct_cam_to_sfm(cam_pts, W)
    
    return sfm_pts




class VolumeReconstructor:
    """
    High-level orchestration class for reconstructing an SFM volume
    from multiple frames.

    This class is responsible for:
        - Managing configuration (e.g., intrinsics)
        - Coordinating frame-wise reconstruction
        - Aggregating results into a point cloud

    """

    def __init__(
        self,
        intrinsics: np.ndarray,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            intrinsics: Camera intrinsic matrix.
            device: Optional torch device for computation.
        """
        self.K = intrinsics
        self.device = device

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def reconstruct_from_directory(
        self,
        frame_dir: str,
    ) -> SFMPointCloud:
        """
        Reconstruct volume from a directory of frame files.

        Args:
            frame_dir: Path containing frame files.

        Returns:
            SFMPointCloud
        """
        frame_path = Path(frame_dir)

        if not frame_path.exists():
            raise FileNotFoundError(f"Directory not found: {frame_dir}")

        sfm_pcd = SFMPointCloud()

        for frame, extrinsics in self._iter_frames(frame_path):
            points = self._reconstruct_single_frame(frame, extrinsics)
            sfm_pcd.add_pts(points)

        return sfm_pcd

    # ------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------

    def _iter_frames(
        self,
        frame_path: Path,
    ) -> Iterable[Tuple[torch.Tensor, np.ndarray]]:
        """
        Lazily iterate over frames.
        This can later be replaced with:
            - multiprocessing
            - streaming
            - dataset loader
        """
        files = sorted(frame_path.glob("*.npz"))

        for file in files:
            sample = np.load(file)
            frame = torch.from_numpy(sample["frame"])

            if self.device is not None:
                frame = frame.to(self.device)

            extrinsics = sample["extrinsics"]

            yield frame, extrinsics

    def _reconstruct_single_frame(
        self,
        frame: torch.Tensor,
        extrinsics: np.ndarray,
    ) -> SFMPoints:
        """
        Perform reconstruction for a single frame.

        This method isolates frame-level reconstruction logic.
        """
        img_pts = ImagePoints.load_from_frame(frame, depth=None)

        # Assumes reconstruct_img_to_sfm is your domain function
        return reconstruct_img_to_sfm(
            img_pts,
            extrinsics,
            self.K,
        )    

