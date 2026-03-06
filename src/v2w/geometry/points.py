"""
v2w.geometry.points

This script includes point modules for each coordinate space.

Author: Ozden Ozel
Created: 2026-01-28
"""


from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field, InitVar
from typing import Union
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import open3d as o3d
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
    coords: torch.Tensor
    covariances: torch.Tensor
    colors: torch.Tensor
    alphas: torch.Tensor
    num_batch: int

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
        
        # Check points' shape    
        self._check_shape()
        

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
        return f"Points(coords: {self.coords.shape}, covariances: {self.covariances.shape}, colors: {self.colors.shape}, alphas: {self.alphas.shape})"

    def __iadd__(self, other: Points):
        if self.num_batch != other.num_batch:
            raise ValueError(f"The number of batches for two point clusters does not match, ({self.num_batch} != {other.num_batch})")
        
        self.coords = torch.cat([self.coords, other.coords], dim=0)
        self.covariances = torch.cat([self.covariances, other.covariances], dim=0)
        self.colors = torch.cat([self.colors, other.colors], dim=0)
        self.alphas = torch.cat([self.alphas, other.alphas], dim=0)
            
        # TODO: Solve point duplications
        
        #self.coords = torch.unique(self.coords, dim=0, sorted=True)
        #self.covariances = torch.unique(self.covariances, dim=0, sorted=True)
        #self.colors = torch.unique(self.colors, dim=0, sorted=True)
        #self.alphas = torch.unique(self.alphas, dim=0, sorted=True)
        
        return self

    def _check_shape(self):
        return (
            len({self.coords.shape[0], 
                 self.covariances.shape[0], 
                 self.colors.shape[0], 
                 self.alphas.shape[0]}) == 1 and
            
            len({self.coords.shape[1], 
                 self.covariances.shape[1], 
                 self.colors.shape[1], 
                 self.alphas.shape[1]}) == 1 and
            
            self.coords.ndim == 3 and
            self.coords.shape[-1] == 3 and

            self.covariances.ndim == 4 and
            self.covariances.shape[-2:] == (3,3) and
            
            self.colors.ndim == 3 and
            self.colors.shape[-1] == 3 and

            self.alphas.ndim == 2
        )
    
    @property
    def bounds(self):
        if len(self) == 0:
            return torch.full((3, 2), float("nan"))
        mins = self.coords.min(dim=0).values
        maxs = self.coords.max(dim=0).values
        return torch.stack([mins, maxs], dim=1)
    

@dataclass
class PointCloud:
    bounds: torch.Tensor
    res: torch.Tensor
    n_downsampling: int
    device: torch.device
    
    def __post_init__(self):
        self._validate_inputs(self.bounds, self.res)

        self.bounds = self.bounds.to(self.device, dtype=torch.float32)
        self.res = self.res.to(self.device, dtype=torch.float32).squeeze()

        self.shape = self._compute_shape()
        self._allocate_storage()

    def _validate_inputs(self, bounds: torch.Tensor, res: torch.Tensor) -> None:
        if bounds.shape != (3, 2):
            raise ShapeError(f"Invalid bounds shape {bounds.shape}, expected (3, 2)")
        if res.shape not in [(3,), (3, 1)]:
            raise ShapeError(f"Invalid resolution shape {res.shape}, expected (3,)")

    def _compute_shape(self) -> torch.Tensor:
        dims = ((self.bounds[:, 1] - self.bounds[:, 0]) / self.res).floor()
        return dims.to(torch.long)

    def _allocate_storage(self) -> None:
        self.coords = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.covariances = torch.empty((0, 3, 3), dtype=torch.float32, device=self.device)
        self.colors = torch.empty((0, 3, 9), dtype=torch.uint8, device=self.device)
        self.alphas = torch.empty((0, ), dtype=torch.float32, device=self.device)

    def add_pts(self, points: Points) -> None:
        # Set the device same as the point cloud
        coords = points.coords.to(self.device)
        covs = points.covariances.to(self.device)
        colors = points.colors.to(self.device)
        alphas = points.alphas.to(self.device)
        
        # Downsample points based on coordinates
        factor = 10 ** self.n_downsampling
        coords = torch.floor(coords * factor) / factor
        unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
        
        indices = torch.arange(len(coords))
        first_indices = torch.full_like(unique_coords, len(coords))

        first_indices = first_indices.scatter_reduce(
            0, inverse, indices, reduce="amin"
        ).sort().values
        
        coords = unique_coords
        covs = covs[first_indices]
        colors = colors[first_indices]
        alphas = alphas[first_indices]
        
        # Compute voxel indices
        indices = ((coords - self.bounds[:, 0]) / self.res).floor().long()
        
        # Remove out-of-bounds points
        mask = ((indices >= 0) & (indices < self.shape)).all(dim=1)
        
        indices = indices[mask]
        coords = coords[mask]
        covs = covs[mask]
        colors = colors[mask]
        alphas = alphas[mask]
        
        # Add new points
        self.coords = torch.cat([self.coords, coords], dim=0)
        self.covariances = torch.cat([self.covariances, covs], dim=0)
        self.colors = torch.cat([self.colors, colors], dim=0)
        self.alphas = torch.cat([self.alphas, alphas], dim=0)
        
        # Remove duplicate points
        self.coords = torch.unique(self.coords, dim=0)
        self.covariances = torch.unique(self.covariances, dim=0)
        self.colors = torch.unique(self.colors, dim=0)
        self.alphas = torch.unique(self.alphas, dim=0)
        
        self.indices = torch.cat([self.indices, indices], dim=0)
        self.indices = torch.unique(self.indices, dim=0, sorted=True)

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
    pass

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
        dtype = torch.float32 if dtype is None else dtype
        
        if self.coords.shape[1] == 0:
            
        # Check points' shape    
        self._check_shape()
    
    def _check_shape(self):
        return (
            len({self.coords.shape[0], 
                 self.covariances.shape[0], 
                 self.colors.shape[0], 
                 self.alphas.shape[0]}) == 1 and
            
            len({self.coords.shape[1], 
                 self.covariances.shape[1], 
                 self.colors.shape[1], 
                 self.alphas.shape[1]}) == 1 and
            
            self.coords.ndim == 3 and
            self.coords.shape[-1] == 2 and

            self.covariances.ndim == 4 and
            self.covariances.shape[-2:] == (2,2) and
            
            self.colors.ndim == 3 and
            self.colors.shape[-1] == 3 and

            self.alphas.ndim == 2
        )
        
    @classmethod
    def _is_empty(tensor: torch.Tensor):
        return tensor.shape[1] == 0
    
    @classmethod
    def load_from_frame(cls, frames: torch.Tensor, depth: torch.Tensor) -> ImagePoints:
        B, H, W = frames[0], frames[2], frames[3]
        
        x = torch.arange(H)
        y = torch.arange(W)
        xy = torch.cartesian_prod(x, y)
        z = depth.flatten().unsqueeze(-1)
        
        N = frames.shape[0] * frames.shape[1]
        
        return ImagePoints(
            coords = torch.cat([xy, z], dim=1).repeat(B, 1),
            covariances = torch.rand(N, 2, 2),
            colors = frames.reshape(-1, 3),
            alphas = torch.rand(N)
        )
            
                
    def scatter(self, step):
        fig = plt.figure()
        ax = fig.add_subplot()

        xs, ys = (self.coords[::step, 0], self.coords[::step, 1])
        rgba = torch.cat([self.colors[::step]/255, self.alphas[::step]], dim=1)
        ax.scatter(xs, ys, s=2, c=rgba)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
        plt.show()

