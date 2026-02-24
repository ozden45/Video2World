"""
gaussian_splatting.py

This script includes an implementation of Gaussian splatting using Pytorch.
"""


import torch
import torch.nn as nn
from v2w.geometry.points import PointCloud
from v2w.geometry.projection import project_sfm_to_cam_tensor, project_cam_to_ray_tensor, project_ray_to_img_tensor
#from v2w.rendering.rasterizer


class PointCloudModel(nn.Module):
    def __init__(self, pts_cloud: PointCloud):
        super().__init__()
        
        # Learnable parameters for each point
        self.coords = nn.Parameter(pts_cloud.coords)
        self.covariances = nn.Parameter(pts_cloud.covariances)
        self.colors = nn.Parameter(pts_cloud.colors)
        self.alphas = nn.Parameter(pts_cloud.alphas)
        
        # Volume index mapping
        x, y, z = torch.meshgrid(
            torch.arange(pts_cloud.shape[0], device='cuda'),
            torch.arange(pts_cloud.shape[1], device='cuda'),
            torch.arange(pts_cloud.shape[2], device='cuda'),
            indexing='ij'
        )
        self.volume_indices = torch.stack([x, y, z], dim=-1)
        

    def forward(self, W: torch.Tensor, K: torch.Tensor):
        # Determine the in-range points
        cam_coords, cam_covariances = project_sfm_to_cam_tensor(self.coords, self.covariances, W)
        mask = (cam_coords[:, 0] > 0) & (cam_coords[:, 1] > 0) & (cam_coords[:, 2] > 0)
        bounds = mask.nonzero(as_tuple=True)
        
        # Determine the bounding box of the in-range points
        x_min, y_min, z_min = bounds.min(dim=0).values
        x_max, y_max, z_max = bounds.max(dim=0).values
        
        # Project the points to the image space
        ray_coords, ray_covariances = project_cam_to_ray_tensor(
            cam_coords[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],
            cam_covariances[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],
        )
        
        img_coords, img_covariances = project_ray_to_img_tensor(
            ray_coords[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],
            ray_covariances[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],
            K
        )
        
        



class GaussianSplattingModel:
    def __init__(self, N: int, cam_params: dict):
        super().__init__()

        fx = cam_params['fx']
        fy = cam_params['fy']
        cx = cam_params['cx']
        cy = cam_params['cy']

        # Camera's intrinsic matrix
        self.K = torch.Tensor([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0, 0, 1]
            ])


        # -------- parameters --------

        # μ (positions)
        self.mu = nn.Parameter(torch.randn(N, 3))

        # covariance 
        self.covariance = nn.Parameter(torch.zeros(N, 3, 3))

        # color (simple RGB for now, not SH)
        self.color = nn.Parameter(torch.rand(N, 3))

        # opacity
        self.alpha = nn.Parameter(torch.ones(N, 1) * 0.5)









import torch
import torch.nn as nn
import torch.nn.functional as F


class PointModel(nn.Module):
    def __init__(self, num_points=1000, feature_dim=32):
        super().__init__()

        # Learnable scene representation
        self.points = nn.Parameter(torch.randn(num_points, 3))
        self.features = nn.Parameter(torch.randn(num_points, feature_dim))

        # Global shared network
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # RGB output example
        )

    def compute_visibility(self, camera_pos):
        """
        Dummy visibility function.
        Replace with real projection/frustum test.
        """
        # distance-based soft visibility
        dist = torch.norm(self.points - camera_pos, dim=1)
        weights = torch.exp(-dist)  # (N,)
        return weights

    def forward(self, camera_pos):
        # 1️⃣ Compute visibility weights
        weights = self.compute_visibility(camera_pos)  # (N,)

        # 2️⃣ Weighted aggregation of features
        weights = weights.unsqueeze(1)  # (N,1)
        aggregated = torch.sum(weights * self.features, dim=0)  # (D,)

        # 3️⃣ Pass through global network
        output = self.mlp(aggregated)

        return output
    