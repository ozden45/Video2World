import pytest
from pathlib import Path
from v2w.geometry.points import *


#====================================================
# Config test fixtures
#====================================================

@pytest.fixture
def cfgs_root_dir():
    return Path(__file__).resolve().parents[1] / "configs"

@pytest.fixture
def cam_cfg_path(cfgs_root_dir):
    return cfgs_root_dir / "cam.yaml"

@pytest.fixture
def dataset_cfg_path(cfgs_root_dir):
    return cfgs_root_dir / "dataset.yaml"

@pytest.fixture
def default_cfg_path(cfgs_root_dir):
    return cfgs_root_dir / "default.yaml"

@pytest.fixture
def model_cfg_path(cfgs_root_dir):
    return cfgs_root_dir / "model.yaml"

@pytest.fixture
def train_cfg_path(cfgs_root_dir):
    return cfgs_root_dir / "train.yaml"



#====================================================
# Geometry test fixtures
#====================================================


@pytest.fixture
def dataset_root_dir():
    return Path(__file__).resolve().parents[1] / "src/v2w/datasets"


@pytest.fixture
def tum_dataset_ext_data_path(dataset_root_dir):
    return dataset_root_dir / "tum_visual_inertial_dataset"


# |---> Points test fixtures

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
            [40, 1, 74]
            ]),
        alphas = torch.tensor([0.5, 0.8])
    )

@pytest.fixture
def pts2():
    return Points(
        coords = torch.tensor([
            [2.3, 0.1, -3], 
            [0.3, 9.9, -8.2]
            ]),
        covariances = torch.tensor([
            [[0.5, 0.3, 0.4], [0.1, 0.1, 0.2], [0.52, 0.13, 0.41]],
            [[0.5, 0.3, 0.4], [0.1, 0.1, 0.2], [0.52, 0.13, 0.41]]
            ]),
        colors = torch.tensor([
            [40, 1, 74], 
            [67, 1, 8]
            ]),
        alphas = torch.tensor([0.3, 0.4])
    )


# |---> Point cloud test fixtures

@pytest.fixture
def bounds():
    return torch.tensor(
        [[-10, 10], [-10, 10], [-10, 10]],
        device=torch.device("cuda")
        )

@pytest.fixture
def res():
    return torch.tensor(
        [0.1, 0.1, 0.1],
        device=torch.device("cuda")
        )

@pytest.fixture
def pts_cloud(bounds, res):
    return PointCloud(bounds, res)


