from __future__ import annotations
import torch
from dataclasses import dataclass, InitVar
import logging
import open3d as o3d



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
        device = self._resolve_device(device)
            
        # Resolve dtype
        dtype = self._resolve_dtype(dtype)
            
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
        # Resolve device
        device = self._resolve_device(device)
            
        # Resolve dtype
        dtype = self._resolve_dtype(dtype)
        
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
        
    @classmethod
    def _is_empty(cls, tensor: torch.Tensor):
        return tensor.shape[1] == 0
        
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
        
    @property
    def bounds(self):
        if len(self) == 0:
            return torch.full((3, 2), float("nan"))
        mins = self.coords.min(dim=0).values
        maxs = self.coords.max(dim=0).values
        return torch.stack([mins, maxs], dim=1)
    
