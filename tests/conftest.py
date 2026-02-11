import pytest
from pathlib import Path

"""
@pytest.fixture(scope="session")
def fixtures_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_image(fixtures_dir):
    return fixtures_dir / "tiny_image.jpg"


@pytest.fixture
def tiny_depth(fixtures_dir):
    return np.load(fixtures_dir / "tiny_depth.npy")


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def random_points():
    return torch.randn(100, 3)
"""



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





