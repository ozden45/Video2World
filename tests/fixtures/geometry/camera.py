import pytest
import torch
from v2w.geometry.points import CamPoints



@pytest.fixture
def extrinsic():
    return torch.tensor([[ 0.70710678, 0.0,  0.70710678, 0.0],
                         [ 0.0,        1.0,  0.0,        0.0], 
                         [-0.70710678, 0.0,  0.70710678, 0.0]], dtype=torch.float64)


@pytest.fixture
def intrinsic():
    return torch.tensor([[800.,   0., 320.],
                         [  0., 800., 240.],
                         [  0.,   0.,   1.]], dtype=torch.float64)
    
    
@pytest.fixture
def cam_pts():
    return CamPoints(
        coords=torch.tensor([[ 4.2426, 2.0000,  2.8284],
                             [ 7.0711, 1.0000,  4.2426],
                             [ 3.5355, 3.0000,  4.9497]], dtype=torch.float64),
        covariances=torch.tensor([[[0.0140, 0.0035,  0.0015],
                                   [0.0035, 0.0150,  0.0020],
                                   [0.0015, 0.0020,  0.0160]],
                                  [[0.0260, -0.0060, 0.0040],
                                   [-0.0060, 0.0250, 0.0050],
                                   [0.0040, 0.0050, 0.0240]],
                                  [[0.0180, 0.0045, 0.0020],
                                   [0.0045, 0.0200, 0.0030],
                                   [0.0020, 0.0030, 0.0190]]], dtype=torch.float64),
        colors=torch.tensor([[121, 10, 204],
                             [1,    2,   3],
                             [0,    0,   0]]),
        alphas=torch.tensor([0.5, 0.8, 0.3])
    )