from pathlib import Path
import os
import numpy as np
import torch
from v2w.datasets.base_dataset import BaseDataset


class TumVIDataset(BaseDataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__(root, split)
        self.root = Path(root) / "split" / "train"
        self.sequences = self._load_split()

    def _load_split(self):
        sequences = []
        for file in os.listdir(self.root):
            if file.endswith(".npz"):
                sequences.append(file[:-4])
                
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sample_path = self.root / f"{sequence}.npz"
        
        data = np.load(sample_path)
        
        return {
            "sequence": data["sequence"],
            "frame": torch.from_numpy(data["frame"]).float(),
            "extrinsics": torch.from_numpy(data["extrinsics"]).float()
        }
        