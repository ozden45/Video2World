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
def W1():
    return torch.tensor(
        [[ 0.70710678, 0.0,  0.70710678, 0.0],
         [ 0.0,        1.0,  0.0,        0.0],
         [-0.70710678, 0.0,  0.70710678, 0.0]],
        dtype=torch.float32
        )
    
@pytest.fixture
def W2():
    return torch.tensor(
        [[1.0,  0.0,       0.0,       0.0],
         [0.0,  0.8660254, -0.5,      -1.0],
         [0.0,  0.5,        0.8660254, 1.0]], 
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
            [[1., 2., 5.], 
             [2., 1., 8.],
             [-1., 3., 6.]]
            ),
        covariances = torch.tensor(
            [[[0.01,  0.003, -0.002],
              [0.003, 0.015,  0.004],
              [-0.002, 0.004, 0.02]],
             [[0.02, -0.005,  0.003],
              [-0.005, 0.025, 0.006],
              [0.003, 0.006, 0.03]],
             [[0.015, 0.004, -0.003],
              [0.004, 0.02,   0.005],
              [-0.003, 0.005, 0.018]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [0, 0, 0]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.3])
    )
    
@pytest.fixture
def cam_pts1():
    return CamPoints(
        coords = torch.tensor(
            [[1., 2., 5.], 
             [7.0711, 1., 4.2426],
             [-1.0, 0.5981, 6.6962]]
            ),
        covariances = torch.tensor(
            [[[0.011, 0.004, -0.003],
              [0.004, 0.016,  0.005],
              [-0.003, 0.005, 0.021]],
             [[0.025, -0.007, 0.005],
              [-0.007, 0.028, 0.008],
              [0.005, 0.008, 0.032]],
             [[0.017, 0.006, -0.004],
              [0.006, 0.023,  0.007],
              [-0.004, 0.007, 0.02]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [0, 0, 0]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.3])
    )

@pytest.fixture
def cam_pts2():
    return CamPoints(
        coords = torch.tensor(
            [[1.0, 2.0, 7.0], 
             [8.0711, 1.0, 4.2426],
             [-1.0, -0.4019, 7.6962]]
            ),
        covariances = torch.tensor(
            [[[0.012, 0.005, -0.004],
              [0.005, 0.017,  0.006],
              [-0.004, 0.006, 0.022]],
             [[0.026, -0.008, 0.006],
              [-0.008, 0.029, 0.009],
              [0.006, 0.009, 0.033]],
             [[0.018, 0.007, -0.005],
              [0.007, 0.024,  0.008],
              [-0.005, 0.008, 0.021]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [0, 0, 0]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.3])
    )

@pytest.fixture
def ray_pts1():
    return RayPoints(
        coords = torch.tensor(
            [[0.1826, 0.3651, 0.9129], 
             [0.8485, 0.1200, 0.5147],
             [-0.1450, 0.0867, 0.9857]]
            ),
        covariances = torch.tensor(
            [[[0.0004,  0.00012, -0.00008],
              [0.00012, 0.0005,   0.00009],
              [-0.00008, 0.00009, 0.0002]],
             [[0.0007, -0.00018, 0.00015],
              [-0.00018, 0.0004, 0.00011],
              [0.00015, 0.00011, 0.0003]],
             [[0.00035, 0.00009, -0.00007],
              [0.00009, 0.00033,  0.00006],
              [-0.00007, 0.00006, 0.00018]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [0, 0, 0]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.3])
    )
    
@pytest.fixture
def ray_pts2():
    return RayPoints(
        coords = torch.tensor(
            [[0.1361, 0.2722, 0.9526], 
             [0.8720, 0.1080, 0.4770],
             [-0.1280, -0.0514, 0.9905]]
            ),
        covariances = torch.tensor(
            [[[0.0003,  0.00010, -0.00006],
              [0.00010, 0.00035,  0.00007],
              [-0.00006, 0.00007, 0.00015]],
             [[0.00065, -0.00017, 0.00014],
              [-0.00017, 0.00035, 0.00010],
              [0.00014, 0.00010, 0.00028]],
             [[0.0003,  0.00008, -0.00006],
              [0.00008, 0.00028,  0.00005],
              [-0.00006, 0.00005, 0.00016]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [0, 0, 0]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.3])
    )
    
@pytest.fixture
def img_pts1():
    return ImagePoints(
        coords = torch.tensor(
            [[480.0, 560.0], 
             [1653.3, 428.6],
             [200.5, 311.5]]
            ),
        covariances = torch.tensor(
            [[[2.6, 0.8],
              [0.8, 3.1]],
             [[7.0, -1.9],
              [-1.9, 4.2]],
             [[2.3, 0.6],
              [0.6, 2.5]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [0, 0, 0]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.3])
    )
    
@pytest.fixture
def img_pts2():
    return ImagePoints(
        coords = torch.tensor(
            [[434.29, 468.57], 
             [1841.2, 428.6],
             [216.1, 198.2]]
            ),
        covariances = torch.tensor(
            [[[1.4, 0.5],
              [0.5, 1.9]],
             [[7.5, -2.1],
              [-2.1, 4.0]],
             [[2.0, 0.55],
              [0.55, 2.2]]]
            ),
        colors = torch.tensor(
            [[121, 10, 204], 
             [1, 2, 3],
             [0, 0, 0]]
            ),
        alphas = torch.tensor([0.5, 0.8, 0.3])
    )
    
