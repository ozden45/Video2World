import torch
import pandas as pd
from pathlib import Path
from typing import Tuple
from v2w.config.loader import load_config
from v2w.utils.misc import if_path_exists
from v2w.utils.math import quat_to_rot_mat


def read_ext_cam_data(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Docstring for read_ext_cam_data
    
    :param path: Description
    :type path: str | Path
    :return: Description
    :rtype: Tuple[Tensor, Tensor]
    """
    
    
    
    # Check the data path exists
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path {path} not found")

    # Initialize time and extrinsic matrix tensor
    ts = torch.tensor([])
    W = torch.tensor([])
    
    # Read .csv file
    df = pd.read_csv(path, encoding="utf-8")
    for row in df.itertuples(index=False):
        ts_i = torch.tensor(row[0] * 1e-9).unsqueeze(-1)
        t_i = torch.tensor(row[1:4]).unsqueeze(-1)
        R_i = quat_to_rot_mat(row[4:])
        W_i = torch.cat([R_i, t_i], dim=1).unsqueeze(0)
        
        ts = torch.cat([ts, ts_i], dim=0)
        W = torch.cat([W, W_i], dim=0)
        
    return ts, W
        
    