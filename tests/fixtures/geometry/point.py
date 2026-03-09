import torch
import pytest
from v2w.geometry.points import Point


@pytest.fixture
def p1():
    return Point(
        coords = torch.tensor([1, 2, 3]),
        covariance = torch.tensor(
            [[0.5, 0.3, 0.4], 
             [0.1, 0.1, 0.2], 
             [0.52, 0.13, 0.41]]
            ),
        color = torch.tensor([121, 10, 204]),
        alpha = torch.tensor([0.5])
    )

@pytest.fixture
def p2():
    return Point(
        coords = torch.tensor([2.3, 0.1, -3]),
        covariance = torch.tensor(
            [[0.5, 0.3, 0.4], 
             [0.1, 0.1, 0.2], 
             [0.52, 0.13, 0.41]]
            ),
        color = torch.tensor([40, 1, 74]),
        alpha = torch.tensor([0.8])
    )