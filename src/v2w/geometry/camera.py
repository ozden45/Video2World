"""
camera.py

Camera geometry and projection utilities
"""


import torch
from typing import Tuple
from dataclasses import dataclass, classmethod
from v2w.config.loader import load_cam_config
from v2w.config.types import CamConfig


def extrinsic_to_view(R: torch.Tensor) -> Tuple[float, float]:
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



class Camera:
    def __init__(self, cfg: CamConfig):
        self.config = cfg
        self.intrinsics = self._set_intrinsics()
        
    def _set_intrinsics(self) -> torch.Tensor:
        
    
    @property
    def intrinsics(self):
        


