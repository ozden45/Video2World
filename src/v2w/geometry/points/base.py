from __future__ import annotations
import torch
from typing import List
from dataclasses import dataclass, InitVar, field
import logging
import open3d as o3d



@dataclass
class Point:
    coords: torch.Tensor
    covariance: torch.Tensor
    color: torch.Tensor
    alpha: torch.Tensor
    
    device: InitVar[torch.device | str | None] = None
    dtype: InitVar[torch.dtype | str | None] = None
    
    def __post_init__(self, device = None, dtype = None):
        # Resolve device
        device = self._resolve_device(device)
            
        # Resolve dtype
        dtype = self._resolve_dtype(dtype)
            
        self.coords = self.coords.to(dtype=dtype, device=device)
        self.covariance = self.covariance.to(dtype=dtype, device=device)
        self.color = self.color.to(dtype=torch.uint8, device=device)
        self.alpha = self.alpha.to(dtype=dtype, device=device)

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
    
    _num_points: int = field(init=False, repr=False)
    
    device: InitVar[torch.device | str | None] = None
    dtype: InitVar[torch.dtype | str | None] = None

    def __post_init__(self, device=None, dtype=None):
        # Resolve device
        device = self._resolve_device(device)
            
        # Resolve dtype
        dtype = self._resolve_dtype(dtype)
        
        # Check points' shape
        self._check_shape()
        
        self.coords = self.coords.to(dtype=dtype, device=device)
        self.covariances = self.covariances.to(dtype=dtype, device=device)
        self.colors = self.colors.to(dtype=torch.uint8, device=device)
        self.alphas = self.alphas.to(dtype=dtype, device=device)
        
        self._num_points = self.alphas.shape[0]
        
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
        self.coords = torch.cat([self.coords, other.coords], dim=0)
        self.covariances = torch.cat([self.covariances, other.covariances], dim=0)
        self.colors = torch.cat([self.colors, other.colors], dim=0)
        self.alphas = torch.cat([self.alphas, other.alphas], dim=0)
            
        self._num_points += self.alphas.shape[0]
            
        # TODO: Solve point duplications
        
        #self.coords = torch.unique(self.coords, dim=0, sorted=True)
        #self.covariances = torch.unique(self.covariances, dim=0, sorted=True)
        #self.colors = torch.unique(self.colors, dim=0, sorted=True)
        #self.alphas = torch.unique(self.alphas, dim=0, sorted=True)
        
        return self

    def _check_shape(self):
        return (
            # Check if num_points dimensions are equal
            len({self.coords.shape[0], 
                 self.covariances.shape[0], 
                 self.colors.shape[0], 
                 self.alphas.shape[0]}) == 1 and
            
            # Check coordinates shape
            self.coords.ndim == 2 and
            self.coords.shape[-1] == 3 and

            # Check covariances shape
            self.covariances.ndim == 3 and
            self.covariances.shape[-2:] == (3,3) and
            
            # Check colors shape
            self.colors.ndim == 2 and
            self.colors.shape[-1] == 3 and

            # Check alphas shape
            self.alphas.ndim == 1
        )
        
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
    def num_points(self):
        return self._num_points
        
    @property
    def bounds(self):
        if len(self) == 0:
            return torch.full((3, 2), float("nan"))
        mins = self.coords.min(dim=0).values
        maxs = self.coords.max(dim=0).values
        return torch.stack([mins, maxs], dim=1)
    


@dataclass
class PointsBatched:
    coords: torch.Tensor
    covariances: torch.Tensor
    colors: torch.Tensor
    alphas: torch.Tensor
    
    _num_batch: int = field(init=False, repr=False)
    _num_points: int = field(init=False, repr=False)
    
    device: InitVar[torch.device | str | None] = None
    dtype: InitVar[torch.dtype | str | None] = None

    def __post_init__(self, device=None, dtype=None):
        # Resolve device
        device = self._resolve_device(device)
            
        # Resolve dtype
        dtype = self._resolve_dtype(dtype)
        
        # Check points' shape
        self._check_shape()
        
        self.coords = self.coords.to(dtype=dtype, device=device)
        self.covariances = self.covariances.to(dtype=dtype, device=device)
        self.colors = self.colors.to(dtype=torch.uint8, device=device)
        self.alphas = self.alphas.to(dtype=dtype, device=device)
        
        self._num_batch = self.alphas.shape[0]
        self._num_points = self.alphas.shape[1]
        
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
            self.coords.shape[-1] == 3 and

            # Check covariances shape
            self.covariances.ndim == 4 and
            self.covariances.shape[-2:] == (3,3) and
            
            # Check colors shape
            self.colors.ndim == 3 and
            self.colors.shape[-1] == 3 and

            # Check alphas shape
            self.alphas.ndim == 2
        )
        
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

    def _is_equal_num_points(self, points: Points):
        return True if self.num_points == points.num_points else False
            

    def add_batch(self, points: Points):
        if not self._is_equal_num_points(points):
            raise ValueError(f"Number of points is not equal, expected {self.num_points}")

        points.coords = points.coords.unsqueeze(0)
        points.covariances = points.covariances.unsqueeze(0)
        points.colors = points.colors.unsqueeze(0)
        points.alphas = points.alphas.unsqueeze(0)
        
        self.coords = torch.cat([self.coords, points.coords], dim=0)
        self.covariances = torch.cat([self.coords, points.covariances], dim=0)
        self.colors = torch.cat([self.coords, points.colors], dim=0)
        self.alphas = torch.cat([self.coords, points.alphas], dim=0)

        self._num_batch += 1

    def extract_all_points(self) -> Points:
        return Points(
            coords=self.coords.reshape(-1, 3),
            covariances=self.covariances.reshape(-1, 3, 3),
            colors=self.colors.reshape(-1, 3),
            alphas=self.alphas.reshape(-1)
        )

    @property
    def num_points(self):
        return self._num_points
    
    @property
    def num_batch(self):
        return self._num_batch
