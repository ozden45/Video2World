"""
Gaussian splatting CUDA extension.

"""

import os
from torch.utils.cpp_extension import load
from pathlib import Path

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

gaussian_splat = gaussian_splat_ext.gaussian_splat
print(type(gaussian_splat))