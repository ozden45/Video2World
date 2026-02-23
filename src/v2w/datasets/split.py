"""
split.py

Dataset split management for Video2World.
"""

from pathlib import Path
import random
from v2w.utils.misc import if_path_exists
from v2w.io import load_frame_csv


def split_tum_vi_dataset(root: str, split_ratios: dict):
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
        split_file = Path(root) / f"tum_vi_{split}.txt"
        with open(split_file) as f:
            sequences = [line.strip() for line in f]
        
        # Here you would implement logic to move/copy files into split-specific directories
        # For example, you could create directories like root/train, root/val, root/test
        # and move the corresponding sequences there.

        sequences_dir = Path(root) / "data"
        csv_path = Path(root) / "data.csv"
        sequences = load_frame_csv(csv_path)
        
        # For randomly splitting the dataset, shuffle the sequences
        random.shuffle(sequences)
        
        # Split the sequences bsaed on the specified ratios
        num_sequences = len(sequences)
        train_end = int(split_ratios["train"] * num_sequences)
        val_end = train_end + int(split_ratios["val"] * num_sequences)
        test_end = val_end + int(split_ratios["test"] * num_sequences)
        
        split_sequences = {
            "train": sequences[:train_end],
            "val": sequences[train_end:val_end],
            "test": sequences[val_end:test_end],
        }
        
        # Create split folder and move files into them
        for split, seqs in split_sequences.items():
            split_dir = Path(root).parents[3] / "splits" / split
            split_dir.mkdir(exist_ok=True)
            
            for seq in seqs:
                seq_path = sequences_dir / seq / ".png"
                
        
