"""
test_splat.py

Unit tests for the Gaussian splatting CUDA extension.
"""


import torch
from v2w.rendering.splat import gaussian_splat_ext
from v2w.geometry.points import SFMPoints
from v2w.exception import ShapeError


def test_gaussian_splat_ext(p1, p2):
    # Create a batch of 3 points
    mu = torch.stack([p1.coords, p2.coords], dim=0)  # (3, 3)
    inv_cov = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # (3, 3, 3)
    clr = torch.tensor([[1.0, 0.0, 0.0], 
                        [0.0, 1.0, 0.0]])  # (2, 3)
    alpha = torch.tensor([0.5, 0.8])  # (2,)
    
    H, W = 480, 640
    
    # Output image buffer
    img = torch.zeros((H, W, 3), dtype=torch.float32, device=torch.device("cuda"))  
    
    # Call the Gaussian splatting extension
    img = gaussian_splat_ext.gaussian_splat(
        img=img,
        mu=mu,
        inv_cov=inv_cov,
        clr=clr,
        alpha=alpha,
        H=H,
        W=W,
        nsigma=20
    )
    
    # Check the output shape
    assert img.shape == (H, W, 3), f"Expected output shape {(H,W,3)}, got {img.shape}"
    #assert False
