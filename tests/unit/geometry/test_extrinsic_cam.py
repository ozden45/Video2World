"""
Docstring for tests.unit.geometry.test_extrinsic_cam
"""

from v2w.geometry.extrinsic_cam import read_ext_cam_data
from pathlib import Path


def test_load_ext_cam_data(tum_dataset_ext_data_path):
    true_path = Path(__file__).resolve().parents[3] / "src/v2w/datasets/tum_visual_inertial_dataset"
    assert true_path == tum_dataset_ext_data_path

    ts, W = read_ext_cam_data(tum_dataset_ext_data_path / "dataset-corridor4_512_16/mav0/mocap0/data.csv")
    assert ts.shape[0] == W.shape[0]


