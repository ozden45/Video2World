import torch
import yaml
import numpy as np
import csv
from pathlib import Path
from .utils import is_path_exists
from .config.types import CameraConfig


    
def load_intrinsic_mat(cfg: CameraConfig) -> torch.Tensor:
    """
    Docstring for compute_int_cam_mat
    
    :return: Description
    :rtype: Tensor
    """

    f_mm = cfg.intrinsic.f_mm
    sensor_width_mm = cfg.intrinsic.sensor_width_mm
    sensor_height_mm = cfg.intrinsic.sensor_height_mm
    width_px = cfg.intrinsic.width_px
    height_px = cfg.intrinsic.height_px
    
    f_x = int((f_mm * width_px) / sensor_width_mm)
    f_y = int((f_mm * height_px) / sensor_height_mm)
    c_x = int(width_px / 2)
    c_y = int(height_px / 2)
    
    return torch.Tensor(
        [[f_x, 0, c_x],
         [0, f_y, c_y],
         [0, 0, 1]]
        )
    

def load_yaml(path: str | Path) -> dict:
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    with open(path) as f:
        return yaml.safe_load(f)


def load_npy(path: str | Path):
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    return np.load(path) 


def read_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        return [row for row in reader if row]

