from pathlib import Path
import cv2 as cv
import torch
import yaml
import numpy as np
import pandas as pd
from typing import List, Tuple
from v2w.utils.misc import if_path_exists


def load_frame(path: str | Path) -> torch.Tensor:
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    frame = cv.imread(path)
    frame = torch.Tensor(frame)
    
    return frame


def load_frame_csv(path: str | Path) -> Tuple[List[str], List[str]]:
    # Check if the file exists
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path '{path}' is not found.")
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Extract the sequence and filename columns as a list
    sequences = df["#timestamp [ns]"].tolist()
    files = df["filename"].tolist()
    
    return sequences, files


def load_extrinsics_csv(path: str | Path) -> Tuple[List[str], torch.Tensor]:
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path '{path}' is not found.")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Extract the sequence and extrinsic columns as a list
    sequences = df["#timestamp [ns]"].tolist()
    translation = df["[p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"].tolist()
    rotation = df["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"].tolist()
    
    return sequences, torch.tensor(translation), torch.tensor(rotation)
    

def load_yaml(path: str | Path) -> dict:
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    with open(path) as f:
        return yaml.safe_load(f)


def load_npy(path: str | Path) -> torch.Tensor:
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    depth = np.load(path)
    depth = torch.Tensor(depth)

