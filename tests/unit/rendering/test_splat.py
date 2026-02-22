"""
test_splat.py

Unit tests for the Gaussian splatting CUDA extension.
"""


import torch
from v2w.geometry.points import ImagePoints
from v2w.rendering.splat import gaussian_splat


def test_gaussian_splat_ext(img_pts: ImagePoints):
    
    H, W = 480, 640
    
    # Output image buffer
    img = torch.zeros((H, W, 3), dtype=torch.float32, device=torch.device("cuda"))  
    
    img = gaussian_splat(img_pts)
    
    # Check the output shape
    assert img.shape == (H, W, 3), f"Expected output shape {(H,W,3)}, got {img.shape}"
    
