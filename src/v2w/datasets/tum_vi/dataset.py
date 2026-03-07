import bisect
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from v2w.utils.math import pose_to_matrix
from v2w.io import read_csv


logger = logging.getLogger(__name__)


class TUMVIDataset(Dataset):
    """
    Returns:
        {
            "images": (2,3,H,W),
            "T_w_c0": (4,4)
        }
    """

    def __init__(
        self,
        root,
        sequence,
        split="train",
        image_size=(512, 512),
    ):
        super().__init__()

        self.root = Path(root)
        self.sequence = sequence
        self.split = split

        logger.info(
            "Initializing TUM-VI dataset | sequence=%s split=%s",
            sequence,
            split
        )

        # -------- Paths --------
        raw_base = self.root / "raw" / sequence / "mav0"
        processed_base = self.root / "processed" / sequence

        self.cam0_dir = raw_base / "cam0" / "data"
        self.cam1_dir = raw_base / "cam1" / "data"

        # -------- Load Split --------
        split_file = processed_base / "splits" / "split.json"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            splits = json.load(f)

        allowed_timestamps = set(splits[split])

        logger.debug("Allowed timestamps: %d", len(allowed_timestamps))

        # -------- Load Images --------
        all_cam0 = sorted(self.cam0_dir.glob("*.png"))
        all_cam1 = sorted(self.cam1_dir.glob("*.png"))

        self.cam0_files = [
            f for f in all_cam0 if f.stem in allowed_timestamps
        ]
        self.cam1_files = [
            f for f in all_cam1 if f.stem in allowed_timestamps
        ]

        assert len(self.cam0_files) == len(self.cam1_files), \
            "cam0 and cam1 frame count mismatch after split"

        if len(self.cam0_files) != len(self.cam1_files):
            raise RuntimeError(
                "cam0 and cam1 frame count mismatch after split"
                f"(cam0={len(self.cam0_files)}, cam1={len(self.cam1_files)})"
            )

        logger.info(
            "Loaded frames | cam0=%d cam1=%d",
            len(self.cam0_files),
            len(self.cam1_files)
        )

        # -------- Load Poses --------
        pose_csv_path = raw_base / "mocap0" / "data.csv"
        
        logger.debug("Loading poses from %s", pose_csv_path)
        
        pose_rows = read_csv(pose_csv_path)

        self.pose_timestamps = []
        self.pose_dict = {}

        skipped_rows = 0

        for row in pose_rows:
            try:
                ts = str(int(float(row[0])))

                if ts not in allowed_timestamps:
                    continue

                px, py, pz = row[1:4]
                qx, qy, qz, qw = row[4:8]

                T_w_c = pose_to_matrix(px, py, pz, qx, qy, qz, qw)

                self.pose_timestamps.append(float(ts))
                self.pose_dict[float(ts)] = T_w_c

            except Exception:
                skipped_rows += 1

        self.pose_timestamps.sort()

        logger.info(
            "Loaded poses | valid=%d skipped=%d",
            len(self.pose_timestamps),
            skipped_rows
        )

        # -------- Transform --------
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        logger.debug("Image transform initialized | size=%s", image_size)

    def __len__(self):
        return len(self.cam0_files)

    def get_pose_nearest(self, timestamp):
        idx = bisect.bisect_left(self.pose_timestamps, timestamp)

        if idx == 0:
            return self.pose_dict[self.pose_timestamps[0]]

        if idx >= len(self.pose_timestamps):
            return self.pose_dict[self.pose_timestamps[-1]]

        before = self.pose_timestamps[idx - 1]
        after = self.pose_timestamps[idx]

        if abs(before - timestamp) < abs(after - timestamp):
            return self.pose_dict[before]
        else:
            return self.pose_dict[after]

    def __getitem__(self, idx):

        file0 = self.cam0_files[idx]
        file1 = self.cam1_files[idx]

        timestamp = float(file0.stem)

        img0 = Image.open(file0).convert("RGB")
        img1 = Image.open(file1).convert("RGB")

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        images = torch.stack([img0, img1], dim=0)

        T_w_c0 = self.get_pose_nearest(timestamp)

        return {
            "images": images,
            "T_w_c0": T_w_c0
        }
        
        