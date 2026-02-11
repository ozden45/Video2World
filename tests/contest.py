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
def configs_path():
    return Path(__file__).resolve().parents[2] / "configs"

@pytest.fixture
def cam_config_path(configs_path):
    return configs_path / "cam.yaml"

@pytest.fixture
def dataset_config_path(configs_path):
    return configs_path / "dataset.yaml"

@pytest.fixture
def default_config_path(configs_path):
    return configs_path / "default.yaml"

@pytest.fixture
def model_config_path(configs_path):
    return configs_path / "model.yaml"

@pytest.fixture
def train_config_path(configs_path):
    return configs_path / "train.yaml"