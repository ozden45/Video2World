"""
gaussian_splatting.py

This script includes an implementation of Gaussian splatting using Pytorch.
"""


import torch
import torch.nn as nn
from v2w.geometry.points import ImagePoint, ImagePoints


class PointModel(nn.Module):
    def __init__(self, img_pts: ImagePoints):
        super().__init__()
        
        # Learnable parameters for each point
        self.coords = nn.Parameter(img_pts.coords) 
        self.covariances = nn.Parameter(img_pts.covariances)
        self.colors = nn.Parameter(img_pts.colors)
        self.alphas = nn.Parameter(img_pts.alphas)


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
    