from pathlib import Path
import cv2 as cv
import torch
import yaml
import numpy as np
from v2w.utils.misc import if_path_exists


def load_image(path: str | Path) -> torch.Tensor:
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    frame = cv.imread(path)
    frame = torch.Tensor(frame)
    
    return frame

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
    