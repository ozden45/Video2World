"""
gaussian_splatting.py

This script includes an implementation of Gaussian splatting using Pytorch.
"""


import torch
import torch.nn as nn
from v2w.geometry.points import ImagePoint, ImagePoints


class PointModel(nn.Module):
    def __init__(self, img_pts: ImagePoint):
        super().__init__()
        




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





