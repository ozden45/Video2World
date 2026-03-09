import torch
import pytest
from v2w.geometry.points import SFMPoints, SFMPointCloud



@pytest.fixture
def sfm_pts():
    return SFMPoints(
        coords=torch.tensor([[ 1., 2., 5.], 
                             [ 2., 1., 8.], 
                             [-1., 3., 6.]], dtype=torch.float64),
        covariances=torch.tensor([[[  0.01,  0.003, -0.002],
                                   [ 0.003,  0.015,  0.004],
                                   [-0.002,  0.004,   0.02]],
                                  [[  0.02, -0.005,  0.003],
                                   [-0.005,  0.025,  .0006],
                                   [ 0.003,  0.006,   0.03]],
                                  [[ 0.015,  0.004, -0.003],
                                   [ 0.004,   0.02,  0.005],
                                   [-0.003,  0.005,  0.018]]], dtype=torch.float64),
        colors=torch.tensor([[121, 10, 204],
                             [  1,  2,   3],
                             [  0,  0,   0]]),
        alphas=torch.tensor([0.5, 0.8, 0.3])
    )


@pytest.fixture
def sfm_pts1():
    return Points(
        coords = torch.tensor([[1,    2,  3], 
                               [2.3, .1, -3]]),
        covariances = torch.tensor([[[ 0.5,  0.3,  0.4], 
                                     [ 0.1,  0.1,  0.2], 
                                     [0.52, 0.13, 0.41]],
                                    [[ 0.5,  0.3,  0.4], 
                                     [ 0.1,  0.1,  0.2], 
                                     [0.52, 0.13, 0.41]]]),
        colors = torch.tensor([[121, 10, 204], 
                               [  1,  2,   3]]),
        alphas = torch.tensor([0.5, 0.8])
    )


@pytest.fixture
def sfm_pts2():
    return Points(
        coords = torch.tensor([[ 4,  5,  6], 
                               [-4, -5, -6]]),
        covariances = torch.tensor([[[ 0.3,  0.3,  0.3], 
                                     [ 0.1,  0.1,  0.2], 
                                     [0.13, 0.13, 0.41]], 
                                    [[ 0.4,  0.4,  0.4], 
                                     [ 0.3,  0.4,  0.9], 
                                     [ 0.5, 0.13, 0.13]]]),
        colors = torch.tensor([[40, 1, 74], 
                               [67, 1,  8]]),
        alphas = torch.tensor([0.3, 0.4])
    )
    
    