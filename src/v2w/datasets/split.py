"""
split.py

Dataset split management for Video2World.
"""

from pathlib import Path
import os
import numpy as np
import torch
import random
from v2w.io import load_frame_as_numpy, load_frame_csv, load_extrinsics_csv
from v2w.utils.misc import if_path_exists
from v2w.utils.math import quat_to_rot_mat


def _save_sample_tum_vi(path: str | Path, frame: torch.Tensor, extrinsics: torch.Tensor):
    np.savez(
        path,
        frame=frame.numpy(),
        extrinsics=extrinsics.numpy()
    )


def split_tum_vi_dataset(root: str, split_dir: str, split_ratios: dict):
    """
    Splits the raw TUM VI dataset into train/val/test sets based on predefined splits.
    
    Args:
        root: Root directory of the TUM VI dataset.
        split_ratios: A dictionary defining the ratios for each split, e.g., {"train": 0.7, "val": 0.15, "test": 0.15}.
    """
    # Check if the dataset root exists
    if not if_path_exists(root):
        raise FileNotFoundError(f"TUM VI dataset root directory '{root}' not found.")
    
    # Define the splits
    splits = ["train", "val", "test"]
    
    for split in splits:
        # Define paths
        cam1_dir = Path(root) / "mav0" / "cam1"
        mocap0_dir = Path(root) / "mav0" / "mocap0"
        img_dir = Path(cam1_dir) / "data"
        frame_csv = Path(cam1_dir) / "data.csv"
        
        # Load sequences and frame names from the CSV file
        seqs, frame_name = load_frame_csv(frame_csv)
        
        # For randomly splitting the dataset, shuffle the sequences
        random.shuffle(seqs)
        
        ext_csv = Path(mocap0_dir) / "data.csv"
        _, translation, rotation_q = load_extrinsics_csv(ext_csv)
        
        # Convert quaternion to rotation matrix
        rotation = quat_to_rot_mat(rotation_q)
        
        # Combine translation and rotation into extrinsics
        extrinsics = torch.cat([
            torch.tensor(rotation),
            torch.tensor(translation)
            ], dim=2)

        # Load frame
        frame_path = img_dir / frame_name
        frame = load_frame_as_numpy(frame_path)

        # Save the sample to the corresponding split directory
        dest_path = Path(split_dir) / split / f"{frame_name}.npz" 
        _save_sample_tum_vi(dest_path, frame, extrinsics)

        
