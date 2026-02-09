# --------------------------------------------------------------
#   extrinsic_cam.py
#
#   Description:
#
#   
#   Author: Ozden Ozel
#   Created: 2026-01-29
#
# --------------------------------------------------------------


import torch
import pandas as pd
from pathlib import Path
from typing import Tuple
from ..config.loader import load_config

def read_ext_cam_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Docstring for read_ext_cam_data
    
    :return: Description
    :rtype: Tuple[Tensor, Tensor]
    """
    
    # Read cam config file
    cam_cfg = load_config()
    data_path = Path(cam_cfg.extrinsic.csv_data_path)
    
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
        
    

def quat_to_rot_mat(q: torch.Tensor | Tuple) -> torch.Tensor:
    """
        Converts a quaternion to a rotation matrix.
    Args:
        q (torch.Tensor): A tensor of shape (4,) representing the quaternion [q0, q1, q2, q3].
    Returns:
        torch.Tensor: A tensor of shape (3, 3) representing the rotation matrix.
    """
    
    q0, q1, q2, q3 = q

    R = torch.Tensor([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, q0**2 - q1**2 + q2**2 - q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0**2 - q1**2 - q2**2 + q3**2]
        ])

    return R


def rot_mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
        Converts a rotation matrixx to a quaternion.
    Args:
        R (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix.
    Returns:
        q (torch.Tensor): A tensor of shape (4,) representing the quaternion [q0, q1, q2, q3].
    """

    q0_abs = 0.5 * torch.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    q1_abs = 0.5 * torch.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
    q2_abs = 0.5 * torch.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
    q3_abs = 0.5 * torch.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])

    if q0_abs >= q1_abs and q0_abs >= q2_abs and q0_abs >= q3_abs:
        q0 = q0_abs
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    elif q1_abs >= q0_abs and q1_abs >= q2_abs and q1_abs >= q3_abs:
        q1 = q1_abs
        q0 = (R[2, 1] - R[1, 2]) / (4 * q1)
        q2 = (R[0, 1] + R[1, 0]) / (4 * q1)
        q3 = (R[0, 2] + R[2, 0]) / (4 * q1)
    elif q2_abs >= q0_abs and q2_abs >= q1_abs and q2_abs >= q3_abs:
        q2 = q2_abs
        q0 = (R[0, 2] - R[2, 0]) / (4 * q2)
        q1 = (R[0, 1] + R[1, 0]) / (4 * q2)
        q3 = (R[1, 2] + R[2, 1]) / (4 * q2)
    else:
        q3 = q3_abs
        q0 = (R[1, 0] - R[0, 1]) / (4 * q3)
        q1 = (R[0, 2] + R[2, 0]) / (4 * q3)
        q2 = (R[1, 2] + R[2, 1]) / (4 * q3)

    q = torch.Tensor([q0, q1, q2, q3])
    return q


if __name__ == "__main__":
    data_path = Path("/home/ozden/repos/Video2World/src/datasets/dataset-corridor4_512_16/mav0/mocap0/data.csv")
    read_ext_cam_data(data_path)
