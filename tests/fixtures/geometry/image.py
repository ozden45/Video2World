import pytest
import torch
from v2w.geometry.points import ImagePoints


@pytest.fixture
def img_pts():
    return ImagePoints(
        coords=torch.tensor([[1520.0, 806.0],
                             [1653.3, 428.6],
                             [ 891.0, 724.5]], dtype=torch.float64),
        covariances=torch.tensor([[[6.20,  1.80],
                                   [1.80,  5.90]],
                                  [[7.00, -1.90],
                                   [-1.90, 4.20]],
                                  [[5.50,  1.60],
                                   [1.60,  5.20]]], dtype=torch.float64),
        colors=torch.tensor([[121, 10, 204],
                             [1,    2,   3],
                             [0,    0,   0]]),
        alphas=torch.tensor([0.5, 0.8, 0.3])
    )