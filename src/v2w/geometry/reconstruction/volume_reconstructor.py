import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import logging
from pathlib import Path
from typing import Tuple, Iterable
import numpy as np
from dataclasses import dataclass
from .img_to_sfm import reconstruct_img_to_sfm
from ..points.sfm import SFMPoints, SFMPointCloud
from ..points.image import ImagePoints
from ...models import MonocularDepthModel



logger = logging.getLogger(__name__)


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
    
    depth_model: MonocularDepthModel = MonocularDepthModel()
    transform: transforms = transforms.Resize((1213, 1546))
    bounds: torch.Tensor = torch.tensor(
        [[0, 10],
         [0, 10],
         [0, 10]]
        )
    res: torch.Tensor = torch.tensor(
        [0.1, 0.1, 0.1]
        )
    n_downsampling: int = 3
    intrinsics: torch.Tensor = torch.tensor(
        [[353.,   0., 600.],
         [  0., 442., 600.],
         [  0.,   0.,   1.]]
        )
    

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
            images = self.transform(batch["images"][:, 0, :, :, :].squeeze(1))
            images = np.array(images)
            
            T_w_c0 = self.transform(batch["T_w_c0"][:, :3, :3])
            logging.debug("Image shape %s", images.shape)
            for image in images:
                depth = self.depth_model(np.transpose(image, axes=(1, 2, 0)))

                sfm_pts = self._reconstruct_single_frame(
                    frame=torch.Tensor(image).permute(1, 2, 0),
                    depth=depth,
                    extrinsics=T_w_c0
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
        img_pts = ImagePoints.load_from_frame(frame)

        return reconstruct_img_to_sfm(
            img_pts,
            extrinsics,
            self.intrinsics,
        )