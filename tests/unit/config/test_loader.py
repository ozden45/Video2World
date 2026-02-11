"""
Docstring for tests.unit.config.test_loader
"""

from v2w.config.loader import *
from pathlib import Path


def test_load_config(cam_cfg_path):
    true_path = Path(__file__).resolve().parents[3] / "configs/cam.yaml"
    assert true_path == cam_cfg_path
    
    cfg = load_config(cam_cfg_path)
    assert type(cfg) == Config

    
def test_load_many_config(cam_cfg_path, dataset_cfg_path, model_cfg_path, train_cfg_path):
    assert True

    
def test_save_config():
    assert True
    
    