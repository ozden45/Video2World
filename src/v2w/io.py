from pathlib import Path
import cv2 as cv
import torch
import yaml
import numpy as np
import pandas as pd
from typing import List, Tuple
from v2w.utils.misc import is_path_exists



def load_frame_as_tensor(path: str | Path) -> torch.Tensor:
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    frame = cv.imread(path)
    frame = torch.Tensor(frame)
    
    return frame


def load_frame_as_numpy(path: str | Path) -> np.ndarray:
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    frame = cv.imread(path)
    frame = np.array(frame)
    
    return frame


def load_frame_csv(path: str | Path) -> Tuple[List[str], List[str]]:
    # Check if the file exists
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path '{path}' is not found.")
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Extract the sequence and filename columns as a list
    sequences = df["#timestamp [ns]"].tolist()
    files = df["filename"].tolist()
    
    return sequences, files


def load_extrinsics_csv(path: str | Path) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path '{path}' is not found.")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Extract the sequence and extrinsic columns as a list
    sequences = df["#timestamp [ns]"].tolist()
    translation = df["[p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"].tolist()
    rotation = df["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"].tolist()
    
    return sequences, translation, rotation
    
    
def load_intrinsic_mat() -> torch.Tensor:
    """
    Docstring for compute_int_cam_mat
    
    :return: Description
    :rtype: Tensor
    """
    from v2w.config.loader import load_config
    
    
    # Read cam config file
    path = Path(__file__).resolve().parents[3] / "configs/cam_config.yaml"
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
    

def load_yaml(path: str | Path) -> dict:
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    with open(path) as f:
        return yaml.safe_load(f)


def load_npy(path: str | Path) -> torch.Tensor:
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    depth = np.load(path)
    depth = torch.Tensor(depth)

