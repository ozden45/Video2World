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
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
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


class Points:
    def __init__(self):
        self.coords = torch.tensor([])
        self.covariances = torch.tensor([])
        self.colors = torch.tensor([])
        self.alphas = torch.tensor([])
        self.num_points = 0
        
        self.bounds = torch.full(size=(3, 2), fill_value=None)
        
    def __len__(self):
        return self.num_points
    
    def __getitem__(self, idx: int) -> Point:
        return Point(
            coords=self.coords[idx, :, :],
            covariance=self.covariances[idx, :, :],
            color=self.colors[idx, :, :],
            alpha=self.alphas[idx, :]
        )

    def __iadd__(self, other: Point | Points):
        match other:
            case Point():
                # Update point parameters
                self.coords = torch.cat([self.coords, other.coords], dim=0)
                self.covariances = torch.cat([self.covariances, other.covariance], dim=0)
                self.colors = torch.cat([self.colors, other.color], dim=0)
                self.alphas = torch.cat([self.alphas, other.alpha], dim=0)
                self.num_points += 1
                
                # Update boundary
                if not self.bounds[0, 0] or self.bounds[0, 0] > other.coords[0]: 
                    self.bounds[0, 0] = other.coords[0]
                if not self.bounds.get[0, 1] or self.bounds[0, 1] < other.coords[0]: 
                    self.bounds[0, 1] = other.coords[0]
                    
                if not self.bounds[1, 0] or self.bounds[1, 0] > other.coords[1]: 
                    self.bounds[1, 0] = other.coords[1]
                if not self.bounds[1, 1] or self.bounds[1, 1] < other.coords[1]: 
                    self.bounds[1, 1] = other.coords[1]
                
                if not self.bounds[2, 0] or self.bounds[2, 0] > other.coords[2]: 
                    self.bounds[2, 0] = other.coords[2]
                if not self.bounds[2, 1] or self.bounds[2, 1] < other.coords[2]: 
                    self.bounds[2, 1] = other.coords[2]
                
            case Points():
                # Update point parameters
                self.coords = torch.cat([self.coords, other.coords], dim=0)
                self.covariances = torch.cat([self.covariances, other.covariances], dim=0)
                self.colors = torch.cat([self.colors, other.colors], dim=0)
                self.alphas = torch.cat([self.alphas, other.alphas], dim=0)
                self.num_points += other.num_points
                
                # Update boundary
                min_coords = torch.min(self.coords, dim=1)
                max_coords = torch.max(self.coords, dim=1)
                
                if not self.bounds[0, 0] or self.bounds[0, 0] > min_coords[0]: 
                    self.bounds[0, 0] = min_coords[0]
                if not self.bounds.get[0, 1] or self.bounds[0, 1] < other.coords[0]: 
                    self.bounds[0, 1] = max_coords[0]
                    
                if not self.bounds[1, 0] or self.bounds[1, 0] > min_coords[1]: 
                    self.bounds[1, 0] = min_coords[1]
                if not self.bounds[1, 1] or self.bounds[1, 1] < other.coords[1]: 
                    self.bounds[1, 1] = max_coords[1]
                
                if not self.bounds[2, 0] or self.bounds[2, 0] > min_coords[2]: 
                    self.bounds[2, 0] = min_coords[2]
                if not self.bounds[2, 1] or self.bounds[2, 1] < other.coords[2]: 
                    self.bounds[2, 1] = max_coords[2]
                
            case _:
                raise TypeError(f"Unsupported type for addition: {type(other)}")

        return self    
    

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
        
    def __getitem__(self, x_idx: int, y_idx: int, z_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.coords[x_idx, y_idx, z_idx, :, :], 
            self.covariances[x_idx, y_idx, z_idx, :, :], 
            self.colors[x_idx, y_idx, z_idx, :, :], 
            self.alphas[x_idx, y_idx, z_idx, :]
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

