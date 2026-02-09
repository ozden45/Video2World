# --------------------------------------------------------------
#   splat.py
#
#   Description:
#
#   Author: Özden Özel
#   Created: 2026-02-08
#
# --------------------------------------------------------------

import os
from torch.utils.cpp_extension import load

this_dir = os.path.dirname(__file__)

gaussian_splat_ext = load(
    name="gaussian_splat_ext",
    sources=[
        os.path.join(this_dir, "splat_binding.cpp"),
        os.path.join(this_dir, "splat_kernel.cu"),
    ],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)

gaussian_splat = gaussian_splat_ext.gaussian_splat
print(type(gaussian_splat))