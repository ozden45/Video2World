import bisect
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from v2w.utils.math import pose_to_matrix
from v2w.io import read_csv


class TUMVIBatchedDataset(BaseDataset):
    """
    Returns:
        {
            "images": (2,3,H,W) stereo tensor
            "T_w_c0": (4,4) world-to-cam0 pose
        }
    """

    def __init__(self, root, sequence, image_size=(512, 512)):
        super().__init__()

        base = Path(root) / sequence / "mav0"

        self.cam0_dir = base / "cam0" / "data"
        self.cam1_dir = base / "cam1" / "data"

        # Sorted image files
        self.cam0_files = sorted(self.cam0_dir.glob("*.png"))
        self.cam1_files = sorted(self.cam1_dir.glob("*.png"))

        assert len(self.cam0_files) == len(self.cam1_files), \
            "cam0 and cam1 frame count mismatch"

        # Load mocap ground-truth
        pose_csv_path = base / "mocap0" / "data.csv"
        pose_rows = read_csv(pose_csv_path)

        self.pose_timestamps = []
        self.pose_dict = {}

        for row in pose_rows:
            try:
                ts = float(row[0])
                px, py, pz = row[1:4]
                qx, qy, qz, qw = row[4:8]

                T_w_c = pose_to_matrix(px, py, pz, qx, qy, qz, qw)

                self.pose_timestamps.append(ts)
                self.pose_dict[ts] = T_w_c

            except:
                continue

        self.pose_timestamps.sort()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # converts to float [0,1]
        ])

    def __len__(self):
        return len(self.cam0_files)

    def get_pose_nearest(self, timestamp):
        """
        Find nearest ground-truth pose to image timestamp.
        """
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

        # Load images
        img0 = Image.open(file0).convert("RGB")
        img1 = Image.open(file1).convert("RGB")

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        # Stack stereo
        images = torch.stack([img0, img1], dim=0)  # (2,3,H,W)

        # Get pose
        T_w_c0 = self.get_pose_nearest(timestamp)

        return {
            "images": images,      # (2,3,H,W)
            "T_w_c0": T_w_c0       # (4,4)
        }


# Dataloader factory
def create_dataloader(root, sequence, batch_size=4, num_workers=4):

    dataset = TUMVIBatchedDataset(root, sequence)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader