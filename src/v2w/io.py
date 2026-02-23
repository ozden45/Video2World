from pathlib import Path
import cv2 as cv
import torch
import yaml
import numpy as np
import pandas as pd
from typing import List
from v2w.utils.misc import if_path_exists


def load_frame(path: str | Path) -> torch.Tensor:
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    frame = cv.imread(path)
    frame = torch.Tensor(frame)
    
    return frame


def load_frame_csv(path: str | Path) -> List[str]:
    # Check if the file exists
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path '{path}' is not found.")
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Extract the 'frame_path' column as a list
    sequences = df["#timestamp [ns]"].tolist()
    return sequences


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

