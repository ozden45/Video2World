"""
Gaussian splatting CUDA extension.

"""

import torch
from torch.utils.cpp_extension import load
import os
from pathlib import Path
from typing import Tuple
from v2w.geometry.points import ImagePoints


if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch

this_dir = os.path.dirname(__file__)

gaussian_splat_ext = load(
    name="gaussian_splat_ext",
    sources=[
        Path(this_dir).parent / "cuda" / "splat_binding.cpp",
        Path(this_dir).parent / "cuda" / "splat_kernel.cu"
    ],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


def gaussian_splat(
    img: torch.Tensor,
    mu: torch.Tensor,
    inv_cov: torch.Tensor,
    clr: torch.Tensor,
    alpha: torch.Tensor,
    img_size: Tuple[int, int],
    nsigma: int = 20
) -> torch.Tensor:
    """
    Gaussian splatting function that calls the CUDA extension.

    Args:
        img (torch.Tensor): Output image buffer of shape (H, W, 3).
        mu (torch.Tensor): Point means of shape (N, 3).
        inv_cov (torch.Tensor): Inverse covariance matrices of shape (N, 3, 3).
        clr (torch.Tensor): Point colors of shape (N, 3).
        alpha (torch.Tensor): Point opacities of shape (N,).
        H (int): Image height.
        W (int): Image width.
        nsigma (int): Number of standard deviations to consider for splatting.

    Returns:
        torch.Tensor: The splatted image of shape (H, W, 3).
    """
    
    mu = img_pts.coords
    inv_cov = img_pts.covariances
    clr = img_pts.colors.float() / 255.0
    alpha = img_pts.alphas.float()
    
    return gaussian_splat_ext.gaussian_splat(
        img=img,
        mu=mu,
        inv_cov=inv_cov,
        clr=clr,
        alpha=alpha,
        H=H,
        W=W,
        nsigma=nsigma
    )
    