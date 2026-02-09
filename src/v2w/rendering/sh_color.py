# --------------------------------------------------------------
#   sh_color.py
#
#   Description:
#       This script includes functions used in the calculation 
#       of spherical harmonics color.
#   
#   Author: �zden �zel
#   Created: 2026-01-31
#
# --------------------------------------------------------------


import torch
from typing import Tuple
import sympy as sp
import math


def sh_color_channel(view: torch.Tensor, c_lm: torch.Tensor, sgd_scale: float = 10.0) -> int:
    """
    Calculate the spherical harmonics lighting coefficients based on the view direction.
    Args:
        view (Tuple[float, float]): A tuple containing the azimuth and elevation angles in radians.
    Returns:
        float: The calculated spherical harmonics lighting coefficients.
    """

    theta, phi = view
    #if theta < 0 or theta > torch.pi:
    #    raise ValueError("Theta must be in [0, pi] and phi must be in [0, 2pi]")
    #if phi < 0 or phi > 2 * torch.pi:
    #    raise ValueError("Phi must be in [0, 2pi]")
    if c_lm.shape != (9,):
        raise ValueError("c_lm must be a tensor of shape (9,1)")

    # Calculate lighting term
    v = view
    Y_lm = torch.Tensor([
        sh(v,0,0), 
        sh(v,1,-1), sh(v,1,0), sh(v,1,1), 
        sh(v,2,-2), sh(v,2,-1), sh(v,2,0), sh(v,2,1), sh(v,2,2)
        ])
    C = (c_lm * Y_lm).sum()
    C = torch.sigmoid(C / sgd_scale) * 255

    return int(C)
    


def sh(view: torch.Tensor, l: int, m: int) -> float:
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


def sh_color(view: torch.Tensor, c_lm_rgb: torch.Tensor) -> torch.Tensor:
    """
    Calculate the spherical harmonics lighting coefficients based on the view direction.
    Args:
        view (Tuple[float, float]): A tuple containing the azimuth and elevation angles in radians.
    Returns:
        float: The calculated spherical harmonics lighting coefficients.
    """
    
    theta, phi = view
    #if theta < 0 or theta > torch.pi:
    #    raise ValueError("Theta must be in [0, pi] and phi must be in [0, 2pi]")
    #if phi < 0 or phi > 2 * torch.pi:
    #    raise ValueError("Phi must be in [0, 2pi]")
    if c_lm_rgb.shape != (3, 9):
        raise ValueError(f"c_lm_rgb must be a tuple of three tensors for R, G, B channels: {c_lm_rgb.shape}")
    if c_lm_rgb[0].shape != (9,) or c_lm_rgb[1].shape != (9,) or c_lm_rgb[2].shape != (9,):
        raise ValueError(f"c_lm must be a tensor of shape (9,1): | {c_lm_rgb[0].shape}, {c_lm_rgb[1].shape}, {c_lm_rgb[2].shape}|")
    
    C_R = sh_color_channel(view, c_lm_rgb[0])
    C_G = sh_color_channel(view, c_lm_rgb[1])
    C_B = sh_color_channel(view, c_lm_rgb[2])

    return (C_R, C_G, C_B)




if __name__ == "__main__":
    view = torch.tensor([torch.pi/4, torch.pi/4])
    c_lm_rgb = torch.rand((3,9))
    sh_rgb = sh_color(view, c_lm_rgb)

    print(f"SH color: {sh_rgb}")