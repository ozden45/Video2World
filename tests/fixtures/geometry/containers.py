import pytest
import torch
from v2w.geometry.points import PointCloud



@pytest.fixture
def sfm_pcd():
    return PointCloud(
        bounds=torch.tensor([0, 10],
                            [0, 10],
                            [0, 10]),
        res=torch.tensor([0.1, 0.1, 0.1]),
        n_downsampling=3,
        device=None,
        dtype=None
    )