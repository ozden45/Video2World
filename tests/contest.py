import pytest
import numpy as np
import torch
from pathlib import Path


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
