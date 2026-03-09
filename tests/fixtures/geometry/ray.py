import pytest
import torch
from v2w.geometry.points import *


@pytest.fixture
def ray_pts():
    return RayPoints(
        coords=torch.tensor([[0.7682, 0.3620, 0.5121],
                             [0.8485, 0.1200, 0.5147],
                             [0.5270, 0.4474, 0.7375]], dtype=torch.float64),
        covariances=torch.tensor([[[ 0.00042,  0.00011, -0.00009],
                                   [ 0.00011,  0.00038,  0.00007],
                                   [-0.00009,  0.00007,  0.00029]],
                                  [[ 0.00065, -0.00018,  0.00015],
                                   [-0.00018,  0.00040,  0.00011],
                                   [ 0.00015,  0.00011,  0.00030]],
                                  [[ 0.00050,  0.00013, -0.00010],
                                   [ 0.00013,  0.00044,  0.00009],
                                   [-0.00010,  0.00009,  0.00033]]], dtype=torch.float64),
        colors=torch.tensor([[121, 10, 204],
                             [1,    2,   3],
                             [0,    0,   0]]),
        alphas=torch.tensor([0.5, 0.8, 0.3])
    )