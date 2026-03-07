import torch
from pathlib import Path
from typing import Tuple
import numpy as np
from ..points.sfm import SFMPoints, SFMPointCloud
from ..points.image import ImagePoints
from .img_to_sfm import reconstruct_img_to_sfm, reconstruct_img_to_sfm_tensor



class VolumeReconstructor:
    """
    High-level orchestration class for reconstructing an SFM volume
    from multiple frames.

    This class is responsible for:
        - Managing configuration (e.g., intrinsics)
        - Coordinating frame-wise reconstruction
        - Aggregating results into a point cloud

    """

    def __init__(
        self,
        intrinsics: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            intrinsics: Camera intrinsic matrix.
            device: Optional torch device for computation.
        """
        self.K = intrinsics
        self.device = device


    def reconstruct_from_directory(
        self,
        frame_dir: str,
    ) -> SFMPointCloud:
        """
        Reconstruct volume from a directory of frame files.

        Args:
            frame_dir: Path containing frame files.

        Returns:
            SFMPointCloud
        """
        frame_path = Path(frame_dir)

        if not frame_path.exists():
            raise FileNotFoundError(f"Directory not found: {frame_dir}")

        sfm_pcd = SFMPointCloud()

        for frame, extrinsics in self._iter_frames(frame_path):
            points = self._reconstruct_single_frame(frame, extrinsics)
            sfm_pcd.add_pts(points)

        return sfm_pcd


    def reconstruct_from_dataset(self):
        raise NotImplementedError
    
    def reconstruct_from_stream(self):
        raise NotImplementedError
    
    
    def _iter_frames(
        self,
        frame_path: Path,
    ) -> Iterable[Tuple[torch.Tensor, np.ndarray]]:
        """
        Lazily iterate over frames.
        This can later be replaced with:
            - multiprocessing
            - streaming
            - dataset loader
        """
        files = sorted(frame_path.glob("*.npz"))

        for file in files:
            sample = np.load(file)
            frame = torch.from_numpy(sample["frame"])

            if self.device is not None:
                frame = frame.to(self.device)

            extrinsics = sample["extrinsics"]

            yield frame, extrinsics

    def _reconstruct_single_frame(
        self,
        frame: torch.Tensor,
        extrinsics: np.ndarray,
    ) -> SFMPoints:
        """
        Perform reconstruction for a single frame.

        This method isolates frame-level reconstruction logic.
        """
        img_pts = ImagePoints.load_from_frame(frame, depth=None)

        # Assumes reconstruct_img_to_sfm is your domain function
        return reconstruct_img_to_sfm(
            img_pts,
            extrinsics,
            self.K,
        )