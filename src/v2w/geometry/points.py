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
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
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
    alphas: torch.Tensor = field(default_factory=lambda: torch.empty((0, 1)))

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
    

@dataclass
class PointCloud:
    def __init__(self, points: Points, bounds: torch.Tensor, res: torch.Tensor):
        if bounds.shape != (3, 2):
            raise ShapeError(f"Invalid bounds shape {bounds.shape}, expecting (3, 2)")
        
        if res.shape != (3, 1):
            raise ShapeError(f"Invalid resolution shape {res.shape}, expecting (3, 1)")
        
        self.bounds = bounds
        self.res = res
        
        x_dim = int((bounds[0, 1] - bounds[0, 0]) / res[0])
        y_dim = int((bounds[1, 1] - bounds[1, 0]) / res[1])
        z_dim = int((bounds[2, 1] - bounds[2, 0]) / res[2])
        
        self.shape = torch.tensor([[x_dim], [y_dim], [z_dim]])
        self.coords = torch.zeros((x_dim, y_dim, z_dim, 3, 1), dtype=float)
        self.covariances = torch.zeros((x_dim, y_dim, z_dim, 3, 3), dtype=float)
        self.colors = torch.zeros((x_dim, y_dim, z_dim, 3, 1), dtype=int)
        self.alphas = torch.zeros((x_dim, y_dim, z_dim, 1), dtype=float)
        
        self.indices = np.array([])
        self._append(points)
        
    def __getitem__(self, x_idx: int, y_idx: int, z_idx: int) -> Point:
        return Point(
            coords=self.coords[x_idx, y_idx, z_idx, :, :], 
            covariance=self.covariances[x_idx, y_idx, z_idx, :, :], 
            color=self.colors[x_idx, y_idx, z_idx, :, :], 
            alpha=self.alphas[x_idx, y_idx, z_idx, :]
        )

    def _append(self, other: Point | Points):
        match other:
            case Point():
                coords = other.coords.squeeze()
                x_idx, y_idx, z_idx = (
                    int((coords[0] - self.bounds[0, 0]) / self.res[0]),
                    int((coords[1] - self.bounds[1, 0]) / self.res[1]),
                    int((coords[2] - self.bounds[2, 0]) / self.res[2]),
                )
                
                self.coords[x_idx, y_idx, z_idx, :, :] = other.coords
                self.covariances[x_idx, y_idx, z_idx, :, :] = other.covariance
                self.colors[x_idx, y_idx, z_idx, :, :] = other.color
                self.alphas[x_idx, y_idx, z_idx, :] = other.alpha
                
                self.indices = np.append(self.indices, (x_idx, y_idx, z_idx))

            case Points():
                for i in range(other.num_points):
                    pts = other[i]
                    x_idx, y_idx, z_idx = (
                        int((pts.coords[0] - self.bounds[0, 0]) / self.res[0]),
                        int((pts.coords[1] - self.bounds[1, 0]) / self.res[1]),
                        int((pts.coords[2] - self.bounds[2, 0]) / self.res[2]),
                    )
                    
                    self.coords[x_idx, y_idx, z_idx, :, :] = pts.coords
                    self.covariances[x_idx, y_idx, z_idx, :, :] = pts.covariance
                    self.colors[x_idx, y_idx, z_idx, :, :] = pts.color
                    self.alphas[x_idx, y_idx, z_idx, :] = pts.alpha
                    
                    self.indices = np.append(self.indices, (x_idx, y_idx, z_idx))
                    
            case _:
                raise TypeError(f"Unsupported type for addition: {type(other)}")

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

