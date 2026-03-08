import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple
import numpy as np
from dataclasses import dataclass
from ..points.sfm import SFMPoints, SFMPointCloud
from ..points.image import ImagePoints
from .img_to_sfm import reconstruct_img_to_sfm



@dataclass
class VolumeReconstructor:
    """
    High-level orchestration class for reconstructing an SFM volume
    from multiple frames.

    This class is responsible for:
        - Managing configuration (e.g., intrinsics)
        - Coordinating frame-wise reconstruction
        - Aggregating results into a point cloud

    """
    
    bounds: torch.Tensor
    res: torch.Tensor
    n_downsampling: int
    intrinsics: torch.Tensor
    

    def _resolve_device(self, device):
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
        
    def _resolve_dtype(self, dtype):
        if dtype is None:
            return torch.float32
        else:
            return torch.as_tensor(1, dtype=dtype).dtype

    def reconstruct_from_directory(
        self,
        frame_dir: str,
        device = None,
        dtype = None
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

        # Resolve device and dtype
        device = self._resolve_device(device)
        dtype = self._resolve_dtype(dtype)

        sfm_pcd = SFMPointCloud(
            bounds=self.bounds,
            res=self.res,
            n_downsampling=self.n_downsampling,
            device=device,
            dtype=dtype
        )

        for frame, extrinsics in self._iter_frames(frame_path):
            points = self._reconstruct_single_frame(frame, extrinsics)
            sfm_pcd.add_pts(points)

        return sfm_pcd


    def reconstruct_from_dataset(
        self,
        loader: DataLoader,
        device = None,
        dtype = None
    ) -> SFMPointCloud:
        """
        Reconstruct volume from a dataset.

        Args:
            loader: Dataloader containing a dataset.
            device: Torch device
            dtype: Torch dtype

        Returns:
            SFMPointCloud
        """
        
        sfm_pcd = SFMPointCloud(
            bounds=self.bounds,
            res=self.res,
            n_downsampling=self.n_downsampling,
            device=self._resolve_device(device),
            dtype=self._resolve_dtype(dtype)
        )
        
        for batch in loader:
            images = batch["images"]
            T_w_c0 = batch["T_w_c0"]
         
            sfm_pts = self._reconstruct_single_frame(
                frame=images[:, 0, :, :, :],
                depth=None,
                extrinsics=T_w_c0[:, :3, :3]
            )
            
            sfm_pcd.add_pts(sfm_pts)
            
        return sfm_pcd
            
    
    def reconstruct_from_stream(self):
        raise NotImplementedError
    
    
    def _reconstruct_single_batch(
        self,
        frame: torch.Tensor,
        depth: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        """
        
        img_pts = ImagePoints.load_from_frame(frame, depth)

        return reconstruct_img_to_sfm(
            img_pts,
            extrinsics,
            self.intrinsics,
        )
    
    
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
        depth: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> SFMPoints:
        """
        Perform reconstruction for a single frame.

        This method isolates frame-level reconstruction logic.
        """
        img_pts = ImagePoints.load_from_frame(frame, depth)

        return reconstruct_img_to_sfm(
            img_pts,
            extrinsics,
            self.intrinsics,
        )