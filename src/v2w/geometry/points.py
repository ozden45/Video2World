"""
v2w.geometry.points

This script includes point modules for each coordinate space.

Author: Ozden Ozel
Created: 2026-01-28
"""


from __future__ import annotations
import torch
from dataclasses import dataclass, field, InitVar
from typing import Union
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from v2w.io import load_frame_as_tensor
from v2w.utils.misc import is_path_exists
from v2w.exception import ShapeError


# --------------------------------------------------------------
# Base classes for point modules
# --------------------------------------------------------------

@dataclass
class Point:
    coords: torch.Tensor       # 3D coordinates
    covariance: torch.Tensor   # Covariance matrix
    color: torch.Tensor        # Color information
    alpha: torch.Tensor        # Opacity
    
    device: InitVar[torch.device | str | None] = None
    dtype: InitVar[torch.dtype | str | None] = None
    
    def __post_init__(self, device=None, dtype=None):
        # Resolve device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            
        # Resolve dtype
        if dtype is None:
            dtype = torch.float32
        else:
            dtype = torch.as_tensor(1, dtype=dtype).dtype
            
        self.coords = self.coords.to(dtype=dtype, device=device)
        self.covariance = self.covariance.to(dtype=dtype, device=device)
        self.color = self.color.to(dtype=torch.uint8, device=device)
        self.alpha = self.alpha.to(dtype=torch.float32, device=device)

    def __eq__(self, other: Point):
        return (
            torch.allclose(self.coords, other.coords, atol=1e-2, rtol=1e-2) and
            torch.allclose(self.covariance, other.covariance, atol=1e-2, rtol=1e-2) and
            torch.equal(self.color, other.color) and
            torch.allclose(self.alpha, other.alpha, atol=1e-2, rtol=1e-2)
        )

    def __repr__(self):
        return f"Point(coords={self.coords}, covariance={self.covariance}, color={self.color}, alpha={self.alpha})"


@dataclass
class Points:
    coords: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3)))
    covariances: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3, 3)))
    colors: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3)))
    alphas: torch.Tensor = field(default_factory=lambda: torch.empty((0)))

    device: InitVar[torch.device | str | None] = None
    dtype: InitVar[torch.dtype | str | None] = None

    def __post_init__(self, device=None, dtype=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            
        # Resolve dtype
        if dtype is None:
            dtype = torch.float32
        else:
            dtype = torch.as_tensor(1, dtype=dtype).dtype
            
        self.coords = self.coords.to(dtype=dtype, device=device)
        self.covariances = self.covariances.to(dtype=dtype, device=device)
        self.colors = self.colors.to(dtype=torch.uint8, device=device)
        self.alphas = self.alphas.to(dtype=torch.float32, device=device)

    def __eq__(self, other: Points):
        return (
            torch.allclose(self.coords, other.coords, atol=1e-2, rtol=1e-2) and
            torch.allclose(self.covariances, other.covariances, atol=1e-2, rtol=1e-2) and
            torch.equal(self.colors, other.colors) and
            torch.allclose(self.alphas, other.alphas, atol=1e-2, rtol=1e-2)
        )

    def __len__(self):
        return self.coords.shape[0]

    def __repr__(self):
        return f"Points(coords={self.coords}, covariances={self.covariances}, colors={self.colors}, alphas={self.alphas})"

    def __iadd__(self, other):
        if isinstance(other, Point):
            self.coords = torch.cat([self.coords, other.coords.unsqueeze(0)], dim=0)
            self.covariances = torch.cat([self.covariances, other.covariance.unsqueeze(0)], dim=0)
            self.colors = torch.cat([self.colors, other.color.unsqueeze(0)], dim=0)
            self.alphas = torch.cat([self.alphas, other.alpha], dim=0)
        elif isinstance(other, Points):
            self.coords = torch.cat([self.coords, other.coords], dim=0)
            self.covariances = torch.cat([self.covariances, other.covariances], dim=0)
            self.colors = torch.cat([self.colors, other.colors], dim=0)
            self.alphas = torch.cat([self.alphas, other.alphas], dim=0)
        else:
            raise TypeError(...)
            
        # TODO: Solve point duplications
        
        #self.coords = torch.unique(self.coords, dim=0, sorted=True)
        #self.covariances = torch.unique(self.covariances, dim=0, sorted=True)
        #self.colors = torch.unique(self.colors, dim=0, sorted=True)
        #self.alphas = torch.unique(self.alphas, dim=0, sorted=True)
        
        return self
    
    @property
    def bounds(self):
        if len(self) == 0:
            return torch.full((3, 2), float("nan"))
        mins = self.coords.min(dim=0).values
        maxs = self.coords.max(dim=0).values
        return torch.stack([mins, maxs], dim=1)
    

class PointCloud:
    def __init__(
        self,
        bounds: torch.Tensor,
        res: torch.Tensor,
        device: Union[str, torch.device] = "cuda"
    ):
        self.device = torch.device(device)

        self._validate_inputs(bounds, res)

        self.bounds = bounds.to(self.device, dtype=torch.float32)
        self.res = res.to(self.device, dtype=torch.float32).squeeze()

        self.shape = self._compute_shape()
        self._allocate_storage()

        # Occupied voxel indices
        self.indices = torch.empty((0, 3), dtype=torch.long, device=self.device)

    def _validate_inputs(self, bounds: torch.Tensor, res: torch.Tensor) -> None:
        if bounds.shape != (3, 2):
            raise ShapeError(f"Invalid bounds shape {bounds.shape}, expected (3, 2)")
        if res.shape not in [(3,), (3, 1)]:
            raise ShapeError(f"Invalid resolution shape {res.shape}, expected (3,)")

    def _compute_shape(self) -> torch.Tensor:
        dims = ((self.bounds[:, 1] - self.bounds[:, 0]) / self.res).floor()
        return dims.to(torch.long)

    def _allocate_storage(self) -> None:
        x_dim, y_dim, z_dim = self.shape.tolist()

        self.coords = torch.zeros(
            (x_dim, y_dim, z_dim, 3),
            dtype=torch.float32,
            device=self.device,
        )

        self.covariances = torch.zeros(
            (x_dim, y_dim, z_dim, 3, 3),
            dtype=torch.float32,
            device=self.device,
        )

        self.colors = torch.zeros(
            (x_dim, y_dim, z_dim, 3),
            dtype=torch.uint8,
            device=self.device,
        )

        self.alphas = torch.zeros(
            (x_dim, y_dim, z_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def add(self, points: Points) -> None:
        coords = points.coords.to(self.device)
        covs = points.covariances.to(self.device)
        colors = points.colors.to(self.device)
        alphas = points.alphas.to(self.device)
        
        # Compute voxel indices
        indices = ((coords - self.bounds[:, 0]) / self.res).floor().long()
        
        # Remove out-of-bounds points
        valid_mask = ((indices >= 0) & (indices < self.shape)).all(dim=1)
        
        indices = indices[valid_mask]
        coords = coords[valid_mask]
        covs = covs[valid_mask]
        colors = colors[valid_mask]
        alphas = alphas[valid_mask]
        
        x_idx, y_idx, z_idx = indices[:, 0], indices[:, 1], indices[:, 2]

        self.coords[x_idx, y_idx, z_idx] = coords
        self.covariances[x_idx, y_idx, z_idx] = covs
        self.colors[x_idx, y_idx, z_idx] = colors
        self.alphas[x_idx, y_idx, z_idx] = alphas

        self.indices = torch.cat([self.indices, indices], dim=0)
        self.indices = torch.unique(self.indices, dim=0, sorted=True)

    def get_filled_voxels(self) -> Points:
        """
        Returns all occupied voxel data as compact tensor.
        """
        if self.indices.shape[0] == 0:
            return None
        
        x, y, z = self.indices.T
        
        return Points(
            coords=self.coords[x, y, z],
            covariances=self.covariances[x, y, z],
            colors=self.colors[x, y, z],
            alphas=self.alphas[x, y, z]
        )

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs, ys, zs = (self.coords[:, 0], self.coords[:, 1], self.coords[:, 2])
        ax.scatter(xs, ys, zs, s=2, c=self.rgba)
    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    
        plt.show()
        
            


# --------------------------------------------------------------
# Point modules for each coordinate space
# --------------------------------------------------------------

class SFMPoint(Point):
    pass


class SFMPoints(Points):
    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs, ys, zs = (self.coords[:, 0], self.coords[:, 1], self.coords[:, 2])
        ax.scatter(xs, ys, zs, s=2, c=self.rgba)
    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    
        plt.show()

    
class SFMPointCloud(PointCloud):
    def _create_voxel_volume_from_video(self, video_path: str):
        # Check if video path exists
        if not is_path_exists(video_path):
            raise FileNotFoundError(f"Video file not found: '{video_path}'.")
        
        path = Path(video_path)
        files = sorted(path.glob("*.png"))
        
        logging.info(f"Loading video frames from '{video_path}'.")
        for file in files:
            frame = load_frame_as_tensor(file)
            



class CamPoint(Point):
    pass


class CamPoints(Points):
    pass


class RayPoint(Point):
    pass


class RayPoints(Points):
    pass


class ImagePoint(Point):
    pass


class ImagePoints(Points):
    def __post_init__(self, device=None, dtype=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            
        # Resolve dtype
        if dtype is None:
            dtype = torch.float32
        else:
            dtype = torch.as_tensor(1, dtype=dtype).dtype
            
        self.coords = torch.empty((0, 2), dtype=dtype, device=device)
        self.covariances = torch.empty((0, 2, 2), dtype=dtype, device=device)
        self.colors = torch.empty((0, 3), dtype=dtype, device=device)
        self.alphas = torch.empty((0), dtype=dtype, device=device)
    
    
    def load_from_frame(self, frame: torch.Tensor, depth: torch.Tensor):
        x = torch.arange(frame.shape[0])
        y = torch.arange(frame.shape[1])
        xy = torch.cartesian_prod(x, y)
        z = depth.flatten().unsqueeze(-1)
        
        N = frame.shape[0]*frame.shape[1]
        self.coords = torch.cat([xy, z], dim=1).unsqueeze(-1)
        self.covariances = torch.rand(N, 2, 2)
        self.colors = frame.reshape(-1, 3)
        self.alphas = torch.rand(N)
                
                
    def scatter(self, step):
        fig = plt.figure()
        ax = fig.add_subplot()

        xs, ys = (self.coords[::step, 0], self.coords[::step, 1])
        rgba = torch.cat([self.colors[::step]/255, self.alphas[::step]], dim=1)
        ax.scatter(xs, ys, s=2, c=rgba)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
        plt.show()

