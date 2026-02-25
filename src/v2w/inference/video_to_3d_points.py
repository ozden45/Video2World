"""
video_to_3d_points.py

This module contains functions for reconstructing 3D points from video frames.
"""


import torch
from v2w.geometry.points import ImagePoints, SFMPoints, SFMPointCloud
from v2w.utils.misc import is_path_exists


def video_to_3d_points(frames_path: str | Path) -> SFMPointCloud:
    """
    Reconstructs 3D points from video frames.
    """

    # Check if the frames path exists
    if not is_path_exists(frames_path):
        raise FileNotFoundError(f"Frames path {frames_path} does not exist.")
    
    # 

