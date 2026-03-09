
pytest_plugins = [
    "tests.fixtures.paths",
    "tests.fixtures.images",
    "tests.fixtures.tensors",
    "tests.fixtures.models",
    "tests.fixtures.datasets",
    "tests.fixtures.geometry",
    "tests.fixtures.config",
]



import pytest
from pathlib import Path
from v2w.geometry.points import *



















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

