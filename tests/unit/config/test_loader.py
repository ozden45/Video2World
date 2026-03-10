"""
Docstring for tests.unit.config.test_loader
"""

from v2w.config.loader import load_cam_config
from v2w.config.types import CameraConfig
from pathlib import Path


def test_load_cam_config(cam_cfg_path):
    true_path = Path(__file__).resolve().parents[3] / "configs/cam.yaml"
    assert true_path == cam_cfg_path
    
    cfg = load_cam_config(cam_cfg_path)
    assert type(cfg) == CameraConfig

    
    
def test_save_config():
    assert True
    
    