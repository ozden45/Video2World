# --------------------------------------------------------------
#   gaussian_splatting.py
#
#   Description:
#       This script includes an implementation of Gaussian 
#       splatting using Pytorch 
#   
#   Author: Özden Özel
#   Created: 2026-01-29
#
# --------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Tuple
from ..geometry.projection import project_sfm_to_img



class PointParameter(nn.Parameter):
    def __init__(self, data=None):
        super(PointParameter, self).__init__()





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




if __name__ == "__main__":
    cam_params = {
        'fx': 2e-3,
        'fy': 2e-3,
        'cx': 640,
        'cy': 320
    }
    
    model = GaussianSplattingModel(1000, cam_params)

    N = 10
    points, covs = project_to_img(
        points=torch.randn((N, 3)), 
        covariances=torch.rand((N, 3, 3)), 
        W=torch.rand((3, 4))
        )

    print("Projected points shape:", points.shape)
    print("Projected covariances shape:", covs.shape)

    #print("Model initialized with parameters:")
    #print("Mu shape:", model.mu.shape)
    #print("Covariance shape:", model.covariance.shape)
    #print("Color shape:", model.color.shape)
    #print("Alpha shape:", model.alpha.shape)



