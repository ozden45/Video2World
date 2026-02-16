# --------------------------------------------------------------
#   points.py
#
#   Description:
#       This script includes point modules for each coordinate 
#       space.
#
#   Author: Ozden Ozel
#   Created: 2026-01-28
#
# --------------------------------------------------------------

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Union
import matplotlib.pyplot as plt
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


@dataclass
class Points:
    coords: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3)))
    covariances: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3, 3)))
    colors: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3)))
    alphas: torch.Tensor = field(default_factory=lambda: torch.empty((0)))

    def __len__(self):
        return self.coords.shape[0]

    @property
    def bounds(self):
        if len(self) == 0:
            return torch.full((3, 2), float("nan"))
        mins = self.coords.min(dim=0).values
        maxs = self.coords.max(dim=0).values
        return torch.stack([mins, maxs], dim=1)

    def __iadd__(self, other):
        if isinstance(other, Point):
            self.coords = torch.cat([self.coords, other.coords.unsqueeze(0)], dim=0)
            self.covariances = torch.cat([self.covariances, other.covariance.unsqueeze(0)], dim=0)
            self.colors = torch.cat([self.colors, other.color.unsqueeze(0)], dim=0)
            self.alphas = torch.cat([self.alphas, other.alpha.unsqueeze(0)], dim=0)
        elif isinstance(other, Points):
            self.coords = torch.cat([self.coords, other.coords], dim=0)
            self.covariances = torch.cat([self.covariances, other.covariances], dim=0)
            self.colors = torch.cat([self.colors, other.colors], dim=0)
            self.alphas = torch.cat([self.alphas, other.alphas], dim=0)
        else:
            raise TypeError(...)
        return self
    

@dataclass(slots=True)
class PointCloud:
    bounds: torch.Tensor
    res: torch.Tensor
    device: torch.device
    
    def __init__(
        self,
        bounds: torch.Tensor,
        res: torch.Tensor,
        device: Union[str, torch.device] = "cuda",
    ):
        self.device = torch.device(device)

        self._validate_inputs(bounds, res)

        self.bounds = bounds.to(self.device, dtype=torch.float32)
        self.res = res.to(self.device, dtype=torch.float32).squeeze()

        self.shape = self._compute_shape()
        self._allocate_storage()

        # store occupied voxel indices (GPU tensor)
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
        
        x_idx, y_idx, z_idx = indices[: 0], indices[: 1], indices[: 2]
        
        self.coords[x_idx, y_idx, z_idx] = coords
        self.covariances[x_idx, y_idx, z_idx] = covs
        self.colors[x_idx, y_idx, z_idx] = colors
        self.alphas[x_idx, y_idx, z_idx] = alphas
        self.indices = torch.cat([self.indices, indices], dim=0)


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
# SFM point modules on the world coordinates
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
    pass



# --------------------------------------------------------------
# Point modules reflected on camera coordinates
# --------------------------------------------------------------

class CamPoint(Point):
    pass


class CamPoints(Points):
    pass


# --------------------------------------------------------------
# Point modules reflected on the ray coordinates
# --------------------------------------------------------------

class RayPoint(Point):
    pass


class RayPoints(Points):
    pass

# --------------------------------------------------------------
# Point modules reflected on image pixel
# --------------------------------------------------------------

class ImagePoint(Point):
    pass


class ImagePoints(Points):
    def __init__(self, frame: torch.Tensor, depth: torch.Tensor):
        x = torch.arange(frame.shape[0])
        y = torch.arange(frame.shape[1])
        xy = torch.cartesian_prod(x, y)
        z = depth.flatten().unsqueeze(-1)
        
        N = frame.shape[0]*frame.shape[1]
        self.coords = torch.cat([xy, z], dim=1).unsqueeze(-1)
        self.covariances = torch.rand(N, 3, 3)
        self.colors = frame.reshape(-1, 3)
        self.alphas = torch.rand(N, 1)
                
    def scatter(self, step):
        fig = plt.figure()
        ax = fig.add_subplot()

        xs, ys = (self.coords[::step, 0], self.coords[::step, 1])
        rgba = torch.cat([self.colors[::step]/255, self.alphas[::step]], dim=1)
        ax.scatter(xs, ys, s=2, c=rgba)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
        plt.show()

