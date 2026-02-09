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
import cv2 as cv


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

    def __len__(self):
        return self.num_points

    def __iadd__(self, other: Point | Points):
        match other:
            case Point():
                self.coords = torch.cat([self.coords, other.coords], dim=0)
                self.covariances = torch.cat([self.covariances, other.covariance], dim=0)
                self.colors = torch.cat([self.colors, other.color], dim=0)
                self.alphas = torch.cat([self.alphas, other.alpha], dim=0)
                self.num_points += 1
            case Points():
                self.coords = torch.cat([self.coords, other.coords], dim=0)
                self.covariances = torch.cat([self.covariances, other.covariances], dim=0)
                self.colors = torch.cat([self.colors, other.colors], dim=0)
                self.alphas = torch.cat([self.alphas, other.alphas], dim=0)
                self.num_points += other.num_points
            case _:
                raise TypeError(f"Unsupported type for addition: {type(other)}")

        return self    
    

class Volume:
    def __init__(self, x_max: int, y_max: int, z_max: int, dx: float, dy: float, dz: float):
        x_dim, y_dim, z_dim = int(x_max / dx), int(y_max / dy), int(z_max / dz)
        self.data = np.full((x_dim, y_dim, z_dim), None, dtype=object)
        
        
        self.indices = np.array([], dtype=torch.Tensor)
        self.delta = (dx, dy, dz)
        
        
        self.coords = torch.tensor((x_dim, y_dim, z_dim), )
        self.covariances = torch.tensor([])
        self.colors = torch.tensor([])
        self.alphas = torch.tensor([])
        

    def __getitem__(self, idx: Tuple[int, int, int]) -> Point:
        return self.data[idx]

    def append(self, other: Point | Points):
        match other:
            case Point():
                coords = other.coords.squeeze()
                index = (
                    int(coords[0] / self.delta[0]),
                    int(coords[1] / self.delta[1]),
                    int(coords[2] / self.delta[2]),
                )
                self.data[index] = other
                self.indices = np.append(self.indices, index)

            case Points():
                for point in other.data:
                    coords = point.coords.squeeze()
                    index = (
                        int(coords[0] / self.delta[0]),
                        int(coords[1] / self.delta[1]),
                        int(coords[2] / self.delta[2]),
                    )
                    self.data[index] = point
                    self.indices = np.append(self.indices, index)

            case _:
                raise TypeError(f"Unsupported type for addition: {type(other)}")
            
    
    @property
    def coords(self) -> torch.Tensor:
        coords = []
        for index in self.indices:
            point = self.data[index]
            if point is None:
                raise ValueError(f"No point found at index {index}")
            coords.append(point.coords.squeeze())

        return torch.stack(coords)

    @property
    def covariances(self) -> torch.Tensor:
        covariances = []
        for index in self.indices:
            point = self.data[index]
            if point is None:
                raise ValueError(f"No point found at index {index}")
            covariances.append(point.covariance)

        return torch.stack(covariances)

    @property
    def colors(self) -> torch.Tensor:
        colors = []
        for index in self.indices:
            point = self.data[index]
            if point is None:
                raise ValueError(f"No point found at index {index}")
            colors.append(point.color)

        return torch.stack(colors)
    
    @property
    def alphas(self) -> torch.Tensor:
        alphas = []
        for index in self.indices:
            point = self.data[index]
            if point is None:
                raise ValueError(f"No point found at index {index}")
            alphas.append(point.alpha)

        return torch.stack(alphas)

    @property
    def rgba(self) -> torch.Tensor:
        rgb = []
        a = []
        for index in self.indices:
            point = self.data[index]
            if point is None:
                raise ValueError(f"No point found at index {index}")
            rgb.append(point.color)
            a.append(point.alpha)

        return torch.cat([rgb, a.unsqueeze(-1)], dim=-1)

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

    
class SFMPointVolume(Volume):
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

