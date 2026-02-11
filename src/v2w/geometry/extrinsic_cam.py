import torch
import pandas as pd
from pathlib import Path
from typing import Tuple
from v2w.config.loader import load_config


def read_ext_cam_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Docstring for read_ext_cam_data
    
    :return: Description
    :rtype: Tuple[Tensor, Tensor]
    """
    
    # Read cam config file
    path = Path("../../../configs/cam.yaml")
    cfg = load_config(path)
    data_path = Path(cfg.camera.extrinsic.csv_data_path)
    
    # Check the data path exists and is correct
    if not data_path.exists():
        raise FileNotFoundError(f"The path '{data_path}' does not exist.")
    elif not data_path.suffix == ".csv":
        raise TypeError(f"The file type {data_path.suffix} is not correct, expecting .csv.")

    # Initialize time and extrinsic matrix tensor
    ts = torch.tensor([])
    W = torch.tensor([])
    
    # Read .csv file
    df = pd.read_csv(data_path, encoding="utf-8")
    for row in df.itertuples(index=False):
        ts_i = torch.tensor(row[0] * 1e-9).unsqueeze(-1)
        t_i = torch.tensor(row[1:4]).unsqueeze(-1)
        R_i = quat_to_rot_mat(row[4:])
        W_i = torch.cat([R_i, t_i], dim=1).unsqueeze(0)
        
        ts = torch.cat([ts, ts_i], dim=0)
        W = torch.cat([W, W_i], dim=0)
        
    return ts, W
        
    