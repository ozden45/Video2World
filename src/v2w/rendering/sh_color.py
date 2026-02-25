"""
sh_color.py

SH color calculation based on view direction and spherical harmonics coefficients.
"""

import torch
import sympy as sp
import math
from typing import Tuple
from v2w.exception import ShapeError


def _sh_color_channel(view: Tuple[float, float], c_lm: torch.Tensor, sgd_scale: float = 10.0) -> int:
    """
    Calculate the spherical harmonics lighting coefficients based on the view direction.
    Args:
        view (Tuple[float, float]): A tuple containing the azimuth and elevation angles in radians.
        c_lm: (torch.Tensor): The spherical harmonics coefficients for single color channel of shape (N, 9).
    Returns:
        float: The calculated spherical harmonics lighting coefficients.
    """

    # Calculate lighting term
    v = view
    Y_lm = torch.Tensor([
        _sh(v, 0, 0), 
        _sh(v, 1, -1), _sh(v, 1, 0), _sh(v, 1, 1), 
        _sh(v, 2, -2), _sh(v, 2, -1), _sh(v, 2, 0), _sh(v, 2, 1), _sh(v, 2, 2)
        ])
    C = torch.einsum('ni,i->n', c_lm, Y_lm)
    C = torch.sigmoid(C / sgd_scale) * 255

    return int(C)
    


def _sh(view: Tuple[float, float], l: int, m: int) -> float:
    """
    Calculate the spherical harmonics basis functions up to the 2nd order.
    Args:
        view (Tuple[float, float]): A tuple containing the azimuth and elevation angles in radians.
    Returns:
        torch.Tensor: A tensor containing the spherical harmonics basis functions.
    """

    theta, phi = view 

    alpha = ((-1)**m) * torch.sqrt(torch.tensor((2*l + 1) / (4*torch.pi) * math.factorial(l - m) / math.factorial(l + m)))
    x = sp.symbols('x')
    expr = sp.assoc_legendre(l, m, x)
    legendre = sp.lambdify(x, expr, 'torch')
    P_lm = legendre(torch.cos(theta))

    exp_term = torch.exp(1j * m * phi)   # requires PyTorch complex support, yields complex tensor
    Y = alpha * P_lm * exp_term
    return float(Y.real)



def sh_color(view: Tuple[float, float], c_lm_rgb: torch.Tensor) -> torch.Tensor:
    """
    Calculate the spherical harmonics lighting coefficients based on the view direction.
    Args:
        view (Tuple[float, float]): A tuple containing the azimuth and elevation angles in radians.
    Returns:
        float: The calculated spherical harmonics lighting coefficients.
    """
    
    if c_lm_rgb.shape[0] != (3, 9):
        raise ShapeError(f"c_lm_rgb must be a tuple of three tensors for R, G, B channels: {c_lm_rgb.shape}")
    
    C_R = _sh_color_channel(view, c_lm_rgb[:, 0, :])
    C_G = _sh_color_channel(view, c_lm_rgb[:, 1, :])
    C_B = _sh_color_channel(view, c_lm_rgb[:, 2, :])

    return (C_R, C_G, C_B)

