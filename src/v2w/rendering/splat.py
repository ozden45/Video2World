"""
Gaussian splatting CUDA extension.

"""

import os
import torch
from torch.utils.cpp_extension import load
from pathlib import Path


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

gaussian_splat = gaussian_splat_ext.gaussian_splat
print(type(gaussian_splat))