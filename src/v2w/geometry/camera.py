"""
camera.py

Camera geometry and projection utilities
"""


import torch
import pandas as pd
from typing import Tuple, Iterable, Optional
from dataclasses import classmethod
from v2w.config.types import CamConfig





class Camera:
    def __init__(self, cfg: CamConfig):
        self.config = cfg
        self.intrinsics = self._set_intrinsics()
        
    def _set_intrinsics(self) -> torch.Tensor:
        f_mm = cfg.camera.intrinsic.f_mm
        sensor_width_mm = cfg.camera.intrinsic.sensor_width_mm
        sensor_height_mm = cfg.camera.intrinsic.sensor_height_mm
        width_px = cfg.camera.intrinsic.width_px
        height_px = cfg.camera.intrinsic.height_px

        f_x = int((f_mm * width_px) / sensor_width_mm)
        f_y = int((f_mm * height_px) / sensor_height_mm)
        c_x = int(width_px / 2)
        c_y = int(height_px / 2)

        return torch.Tensor(
            [[f_x, 0, c_x],
             [0, f_y, c_y],
             [0, 0, 1]]
            )

    def extrinsics_stream(path: str, dataset: Optional[str | None] = None) -> Iterable[Tuple[torch.Tensor, np.ndarray]]:
        """
        Load extrinsic camera parameters from specified dataset.
        Args:
            path (str): Path for extrinsic camera parameters.
            dataset (Optional[str | None]):
        Returns:

        """
        
        if not is_path_exists(path):
            raise FileNotFoundError(f"The path '{path}' is not found.")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(path)
    
    # Extract the sequence and extrinsic columns as a list
    sequences = df["#timestamp [ns]"].tolist()
    translation = df[[
        ' p_RS_R_x [m]', 
        ' p_RS_R_y [m]', 
        ' p_RS_R_z [m]']].to_numpy().tolist()
    rotation = df[[
        ' q_RS_w []', 
        ' q_RS_x []', 
        ' q_RS_y []', 
        ' q_RS_z []']].to_numpy().tolist()
    
    return sequences, translation, rotation
        
        
        

    @classmethod
    def extrinsic_to_view(cls, R: torch.Tensor) -> Tuple[float, float]:
        """
        Convert extrinsic camera parameters (rotation and translation) to view direction (azimuth and elevation).
        Args:
            R (torch.Tensor): Rotation matrix of shape (3, 3).
        Returns:
            Tuple[float, float]: A tuple containing the azimuth and elevation angles in radians.
        """
        # Assuming the camera looks along the negative z-axis in its local coordinate system
        view_dir = -R[:, 2]  # The third column of R gives the forward direction
        azimuth = torch.atan2(view_dir[0], view_dir[2])  # atan2(x, z)
        elevation = torch.asin(view_dir[1])  # asin(y)
    
        return azimuth.item(), elevation.item()


    
    
    @property
    def intrinsics(self):
        return self.intrinsics


