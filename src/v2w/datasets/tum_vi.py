from pathlib import Path
from .base_dataset import BaseDataset


class TumVIDataset(BaseDataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__(root, split)
        self.root = Path(root)
        self.sequences = self._load_split()

    def _load_split(self):
        split_file = Path("data/splits") / f"tum_vi_{self.split}.png"
        with open(split_file) as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        seq_path = self.root / sequence
        
        
        
        
        # Load image and extrinsiics
        
        return {
            "image": image,
            "extrinsics": extrinsics,
            "sequence": sequence
            }
    