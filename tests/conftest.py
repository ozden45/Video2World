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
        [[ 0.70710678, 0.0,  0.70710678, 0.0],
         [ 0.0,        1.0,  0.0,        0.0],
         [-0.70710678, 0.0,  0.70710678, 0.0]],
        dtype=torch.float64
    )


@pytest.fixture
def K():
    return torch.tensor(
        [[800., 0., 320.],
         [0., 800., 240.],
         [0., 0., 1.]],
        dtype=torch.float64
    )


@pytest.fixture
def sfm_pts():
    return SFMPoints(
        coords=torch.tensor(
            [[1., 2., 5.],
             [2., 1., 8.],
             [-1., 3., 6.]],
            dtype=torch.float64
        ),
        covariances=torch.tensor(
            [[[0.01,  0.003, -0.002],
              [0.003, 0.015,  0.004],
              [-0.002, 0.004, 0.02]],
             [[0.02, -0.005,  0.003],
              [-0.005, 0.025, 0.006],
              [0.003, 0.006, 0.03]],
             [[0.015, 0.004, -0.003],
              [0.004, 0.02,   0.005],
              [-0.003, 0.005, 0.018]]],
            dtype=torch.float64
        ),
        colors=torch.tensor([[121,10,204],[1,2,3],[0,0,0]]),
        alphas=torch.tensor([0.5,0.8,0.3])
    )


@pytest.fixture
def cam_pts():
    return CamPoints(
        coords=torch.tensor(
            [[ 4.2426, 2.0000,  2.8284],
             [ 7.0711, 1.0000,  4.2426],
             [ 3.5355, 3.0000,  4.9497]],
            dtype=torch.float64
        ),
        covariances=torch.tensor(
            [[[0.0140, 0.0035,  0.0015],
              [0.0035, 0.0150,  0.0020],
              [0.0015, 0.0020,  0.0160]],
             [[0.0260, -0.0060, 0.0040],
              [-0.0060, 0.0250, 0.0050],
              [0.0040, 0.0050, 0.0240]],
             [[0.0180, 0.0045, 0.0020],
              [0.0045, 0.0200, 0.0030],
              [0.0020, 0.0030, 0.0190]]],
            dtype=torch.float64
        ),
        colors=torch.tensor([[121,10,204],[1,2,3],[0,0,0]]),
        alphas=torch.tensor([0.5,0.8,0.3])
    )


@pytest.fixture
def ray_pts():
    return RayPoints(
        coords=torch.tensor(
            [[0.7682, 0.3620, 0.5121],
             [0.8485, 0.1200, 0.5147],
             [0.5270, 0.4474, 0.7375]],
            dtype=torch.float64
        ),
        covariances=torch.tensor(
            [[[0.00042,  0.00011, -0.00009],
              [0.00011,  0.00038,  0.00007],
              [-0.00009, 0.00007,  0.00029]],
             [[0.00065, -0.00018, 0.00015],
              [-0.00018, 0.00040, 0.00011],
              [0.00015, 0.00011, 0.00030]],
             [[0.00050, 0.00013, -0.00010],
              [0.00013, 0.00044,  0.00009],
              [-0.00010,0.00009,  0.00033]]],
            dtype=torch.float64
        ),
        colors=torch.tensor([[121,10,204],[1,2,3],[0,0,0]]),
        alphas=torch.tensor([0.5,0.8,0.3])
    )


@pytest.fixture
def img_pts():
    return ImagePoints(
        coords=torch.tensor(
            [[1520.0,  806.0],
             [1653.3,  428.6],
             [ 891.0,  724.5]],
            dtype=torch.float64
        ),
        covariances=torch.tensor(
            [[[6.20,  1.80],
              [1.80,  5.90]],
             [[7.00, -1.90],
              [-1.90, 4.20]],
             [[5.50,  1.60],
              [1.60,  5.20]]],
            dtype=torch.float64
        ),
        colors=torch.tensor([[121,10,204],[1,2,3],[0,0,0]]),
        alphas=torch.tensor([0.5,0.8,0.3])
    )
    


#====================================================
# Rendering fixtures
#====================================================

@pytest.fixture
def img_empty():
    H, W = 480, 640
    return torch.zeros(
        (H, W, 3), 
        dtype=torch.float32, 
        device=torch.device("cuda")
        )

