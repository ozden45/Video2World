# --------------------------------------------------------------
#   intrinsic_cam.py
#
#   Description:
#
#   
#   Author: Ozden Ozel
#   Created: 2026-02-08
#
# --------------------------------------------------------------


import torch
from pathlib import Path
from v2w.config.loader import load_config


def compute_int_cam_mat() -> torch.Tensor:
    """
    Docstring for compute_int_cam_mat
    
    :return: Description
    :rtype: Tensor
    """
    
    # Read cam config file
    path = Path("../../../configs/")
    cfg = load_config(path)

    f_mm = cfg.camera.intrinsic.f_mm
    sensor_width_mm = cfg.camera.intrinsic.sensor_width_mm
    sensor_height_mm = cfg.camera.intrinsic.sensor_height_mm
    width_px = cfg.camera.intrinsic.width_px
    height_px = cfg.camera.intrinsic.height_px
    
    f_x = int((f_mm * width_px) / sensor_width_mm)
    f_y = int((f_mm * height_px) / sensor_height_mm)
    c_x = int(width_px / 2)
    c_y = int(height_px / 2)
    
    K = torch.Tensor([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
        ])
    
    return K
    