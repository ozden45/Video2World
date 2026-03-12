from __future__ import annotations
import torch
import matplotlib.pyplot as plt
import logging
import open3d as o3d
from .base import Point, Points, PointsBatched
from ...exception import ShapeError


logger = logging.getLogger(__name__)


class ImagePoint(Point):
    pass


class ImagePoints(Points):
    def _check_shape(self):
        return (
            len({self.coords.shape[0], 
                 self.covariances.shape[0], 
                 self.colors.shape[0], 
                 self.alphas.shape[0]}) == 1 and
            
            self.coords.ndim == 3 and
            self.coords.shape[-1] == 2 and

            self.covariances.ndim == 4 and
            self.covariances.shape[-2:] == (2,2) and
            
            self.colors.ndim == 3 and
            self.colors.shape[-1] == 3 and

            self.alphas.ndim == 2
        )
    
    @classmethod
    def load_from_frame(cls, frame: torch.Tensor) -> ImagePoints:
        
        logger.debug("Loaded frame with shape %s", frame.shape)
        
        H, W = int(frame.shape[0]), int(frame.shape[1])
        
        x = torch.arange(H)
        y = torch.arange(W)
        xy = torch.cartesian_prod(x, y)
        
        N = frame.shape[0] * frame.shape[1]
        
        logger.debug("coords.shape: %s", xy.shape)
        logger.debug("covs.shape: %s", torch.rand(N, 2, 2).shape)
        logger.debug("colors.shape: %s", frame.reshape(-1, 3).shape)
        logger.debug("alphas.shape: %s", torch.rand(N).shape)
        
        return ImagePoints(
            coords = xy,
            covariances = torch.rand(N, 2, 2),
            colors = frame.reshape(-1, 3),
            alphas = torch.rand(N)
        )
            
    def scatter(self, step):
        fig = plt.figure()
        ax = fig.add_subplot()

        xs, ys = (self.coords[::step, 0], self.coords[::step, 1])
        rgba = torch.cat([self.colors[::step] / 255, self.alphas[::step]], dim=1)
        ax.scatter(xs, ys, s=2, c=rgba)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
        plt.show()
        
    

class ImagePointsBatched(PointsBatched):
    def _check_shape(self):
        return (
            # Check if num_batch dimensions are equal
            len({self.coords.shape[0], 
                 self.covariances.shape[0], 
                 self.colors.shape[0], 
                 self.alphas.shape[0]}) == 1 and
            
            # Check if num_points dimensions are equal
            len({self.coords.shape[1], 
                 self.covariances.shape[1], 
                 self.colors.shape[1], 
                 self.alphas.shape[1]}) == 1 and
            
            # Check coordinates shape
            self.coords.ndim == 3 and
            self.coords.shape[-1] == 2 and

            # Check covariances shape
            self.covariances.ndim == 4 and
            self.covariances.shape[-2:] == (2,2) and
            
            # Check colors shape
            self.colors.ndim == 3 and
            self.colors.shape[-1] == 3 and

            # Check alphas shape
            self.alphas.ndim == 2
        )
    
    def extract_all_points(self) -> ImagePoints:
        return ImagePoints(
            coords=self.coords.reshape(-1, 2),
            covariances=self.covariances.reshape(-1, 2, 2),
            colors=self.colors.reshape(-1, 3),
            alphas=self.alphas.reshape(-1)
        )
    
