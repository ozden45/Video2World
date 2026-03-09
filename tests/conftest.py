



import pytest
from pathlib import Path
from v2w.geometry.points import *









#====================================================
# Geometry test fixtures
#====================================================


@pytest.fixture
def dataset_root_dir():
    return Path(__file__).resolve().parents[1] / "src/v2w/datasets"


@pytest.fixture
def tum_dataset_ext_data_path(dataset_root_dir):
    return dataset_root_dir / "tum_visual_inertial_dataset"



# |---> Point test fixtures

@pytest.fixture
def p1():
    return Point(
        coords = torch.tensor([1, 2, 3]),
        covariance = torch.tensor([[0.5, 0.3, 0.4], [0.1, 0.1, 0.2], [0.52, 0.13, 0.41]]),
        color = torch.tensor([121, 10, 204]),
        alpha = torch.tensor([0.5])
    )

@pytest.fixture
def p2():
    return Point(
        coords = torch.tensor([2.3, 0.1, -3]),
        covariance = torch.tensor([[0.5, 0.3, 0.4], [0.1, 0.1, 0.2], [0.52, 0.13, 0.41]]),
        color = torch.tensor([40, 1, 74]),
        alpha = torch.tensor([0.8])
    )



# |---> Points test fixtures

@pytest.fixture
def pts1():
    return Points(
        coords = torch.tensor([
            [1, 2, 3], 
            [2.3, 0.1, -3]
            ]),
        covariances = torch.tensor([
            [[0.5, 0.3, 0.4], [0.1, 0.1, 0.2], [0.52, 0.13, 0.41]],
            [[0.5, 0.3, 0.4], [0.1, 0.1, 0.2], [0.52, 0.13, 0.41]]
            ]),
        colors = torch.tensor([
            [121, 10, 204], 
            [1, 2, 3]
            ]),
        alphas = torch.tensor([0.5, 0.8])
    )

@pytest.fixture
def pts2():
    return Points(
        coords = torch.tensor([
            [4, 5, 6], 
            [-4, -5, -6]
            ]),
        covariances = torch.tensor([
            [[0.3, 0.3, 0.3], [0.1, 0.1, 0.2], [0.13, 0.13, 0.41]],
            [[0.4, 0.4, 0.4], [0.3, 0.4, 0.9], [0.5, 0.13, 0.13]]
            ]),
        colors = torch.tensor([
            [40, 1, 74], 
            [67, 1, 8]
            ]),
        alphas = torch.tensor([0.3, 0.4])
    )




