"""
split.py

Dataset split management for Video2World.
"""

import torch
from pathlib import Path
import numpy as np
from ...io import load_frame_as_tensor, load_tum_vi_frame_csv, load_tum_vi_extrinsics_csv
from ...utils import is_path_exists, quat_to_rot


def _save_sample_tum_vi(
    path: str, 
    sequence: str, 
    frame: torch.Tensor, 
    depth: torch.Tensor,
    extrinsics: torch.Tensor
):
    np.savez(
        path,
        sequence=sequence,
        frame=frame.numpy(),
        depth=depth.numpy(),
        extrinsics=extrinsics.numpy()
    )

def _reduce_timestamps(ts: np.ndarray) -> np.ndarray:
    for i in range(len(ts)):
        ts[i] = ts[i] / 1e12
        
    return ts

def _match_timestamps(ts1: np.ndarray, ts2: np.ndarray) -> np.ndarray:
    pass

def _nearest_match(t1, t2):
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)

    indices = np.searchsorted(t2, t1)

    indices = np.clip(indices, 1, len(t2) - 1)

    left = t2[indices - 1]
    right = t2[indices]

    indices -= (t1 - left < right - t1)

    return indices


def split_tum_vi_dataset(
    root: str, 
    split_dir: str, 
    split_ratios: dict
):
    """
    Splits the raw TUM VI dataset into train/val/test sets based on predefined splits.
    
    Args:
        root: Root directory of the TUM VI dataset.
        split_ratios: A dictionary defining the ratios for each split, e.g., {"train": 0.7, "val": 0.15, "test": 0.15}.
    """
    # Check if the dataset root exists
    if not is_path_exists(root):
        raise FileNotFoundError(f"TUM VI dataset root directory '{root}' not found.")
    
    # Define the splits
    splits = ["train", "val", "test"]
    
    # Define paths
    cam1_dir = Path(root) / "mav0" / "cam1"
    mocap0_dir = Path(root) / "mav0" / "mocap0"
    img_dir = Path(cam1_dir) / "data"
    frame_csv = Path(cam1_dir) / "data.csv"
    
    # Load sequences and frame names from the CSV file
    frame_seqs, frame_names = load_tum_vi_frame_csv(frame_csv)

    # Load extrinsics from the mocap CSV file    
    ext_csv = Path(mocap0_dir) / "data.csv"
    ext_seqs, translation, rotation_q = load_tum_vi_extrinsics_csv(ext_csv)
    
    # Get split indices based on the provided ratios
    N = len(frame_names)
    train_end = int(N * split_ratios["train"])
    val_end = train_end + int(N * split_ratios["val"])
    test_end = val_end + int(N * split_ratios["test"])
    
    for split in splits:
        # Create split directory if it doesn't exist
        split_path = Path(split_dir) / split
        split_path.mkdir(parents=True, exist_ok=True)
        
        # Determine the indices for the current split
        match split:
            case "train":
                split_iter = zip(seqs[:train_end], frame_names[:train_end], translation[:train_end], rotation_q[:train_end])
            case "val":
                split_iter = zip(seqs[train_end:val_end], frame_names[train_end:val_end], translation[train_end:val_end], rotation_q[train_end:val_end])
            case "test":
                split_iter = zip(seqs[val_end:test_end], frame_names[val_end:test_end], translation[val_end:test_end], rotation_q[val_end:test_end])
            
        for seq, frame_name, tr, rot_q in split_iter:
            # Convert quaternion to rotation matrix
            rot = quat_to_rot(rot_q)
                        
            # Combine translation and rotation into extrinsics
            rot_tensor = torch.tensor(rot)
            tr_tensor = torch.tensor(tr).unsqueeze(-1)
            extrinsics = torch.cat([rot_tensor, tr_tensor], dim=1)
            
            # Load frame
            frame_path = img_dir / frame_name
            frame = load_frame_as_tensor(frame_path)

            # Save the sample to the corresponding split directory
            dest_path = Path(split_dir) / split / f"{frame_name}.npz" 
            _save_sample_tum_vi(dest_path, seq, frame, extrinsics)    
        




import json
import random
from pathlib import Path


def create_split(
    root,
    sequence,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    save_file=True
):
    """
    Creates train/val/test splits for a TUMVI sequence.

    Returns dict:
    {
        "train": [...indices...],
        "val": [...indices...],
        "test": [...indices...]
    }
    """

    base = Path(root) / sequence / "mav0"
    cam0_dir = base / "cam0" / "data"

    files = sorted(cam0_dir.glob("*.png"))
    n = len(files)

    indices = list(range(n))

    random.seed(seed)
    random.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:]
    }

    if save_file:
        out_path = Path(root) / sequence / "split.json"
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)

        print(f"Split file saved to: {out_path}")

    return splits


if __name__ == "__main__":

    root = "/path/to/TUMVI"
    sequence = "room1"

    splits = create_split(root, sequence)

    print("Train:", len(splits["train"]))
    print("Val:", len(splits["val"]))
    print("Test:", len(splits["test"]))


