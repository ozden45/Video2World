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


# |---> Projection test fixtures

@pytest.fixture
def W():
    return torch.tensor(
        [[0, 0, 1, 0], 
         [0, 1, 0, 0], 
         [-1, 0, 0, -5]], 
        dtype=torch.float32
        )

@pytest.fixture
def K():
    return torch.tensor(
        [[800, 0, 320], 
         [0, 800, 240], 
         [0, 0, 1]],
        dtype=torch.float32)

@pytest.fixture
def sfm_pts():
    return SFMPoints(
        coords = torch.tensor(
            [[5, 2, 10], 
             [6, 1, 8],
             [4, -1, 12]]
            ),
        covariances = torch.tensor(
            [[[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]],
             [[2, 0, 0], 
              [0, 1, 0], 
              [0, 0, 0.5]],
             [[0.5, 0, 0], 
              [0, 0.5, 0], 
              [0, 0, 0.5]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [40, 1, 74]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.2])
    )
    
@pytest.fixture
def cam_pts():
    return CamPoints(
        coords = torch.tensor(
            [[10, 2, -10], 
             [8, 1, -11],
             [12, -1, -9]]
            ),
        covariances = torch.tensor(
            [[[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]],
             [[0.5, 0, 0], 
              [0, 1, 0], 
              [0, 0, 2]],
             [[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [40, 1, 74]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.2])
    )

@pytest.fixture
def ray_pts():
    return RayPoints(
        coords = torch.tensor(
            [[-1, -0.2, 1], 
             [-0.727, -0.091, 1],
             [-1.333, 0.111, 1]]
            ),
        covariances = torch.tensor(
            [[[0.5, 0.3, 0.4], 
              [0.1, 0.1, 0.2], 
              [0.52, 0.13, 0.41]],
             [[0.5, 0.3, 0.4], 
              [0.1, 0.1, 0.2], 
              [0.52, 0.13, 0.41]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [40, 1, 74]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.2])
    )
    
@pytest.fixture
def img_pts():
    return ImagePoints(
        coords = torch.tensor([
            [-480, 80, 1], 
            [-261.82, 167.27, 1],
            [-746.67, 328.89, 1]
            ]),
        covariances = torch.tensor(
            [[[0.5, 0.3, 0.4], 
              [0.1, 0.1, 0.2], 
              [0.52, 0.13, 0.41]],
             [[0.5, 0.3, 0.4], 
              [0.1, 0.1, 0.2], 
              [0.52, 0.13, 0.41]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [40, 1, 74]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.2])
    )
    
    
    
"""
| Point | Σ_world       | Σ_camera     | Σ_ray (2×2)                            | Σ_image (2×2)                |
| ----- | ------------- | ------------ | -------------------------------------- | ---------------------------- |
| P1    | I             | I            | [[0.02,0.002],[0.002,0.0104]]          | [[12800,1280],[1280,6656]]   |
| P2    | diag(2,1,0.5) | rotated diag | [[0.01167,0.00109],[0.00109,0.00835]]  | [[7469,699],[699,5341]]      |
| P3    | correlated    | rotated      | [[0.0328,-0.00365],[-0.00365,0.01245]] | [[20996,-2339],[-2339,7967]] |

"""