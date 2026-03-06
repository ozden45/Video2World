from __future__ import annotations
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
import logging
import open3d as o3d
from v2w.exception import ShapeError



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
        