from .base import Point, Points
from __future__ import annotations
import torch
import matplotlib.pyplot as plt
import logging
import open3d as o3d
from v2w.exception import ShapeError



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