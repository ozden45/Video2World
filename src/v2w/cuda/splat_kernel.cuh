#pragma once

// Host launcher (normal C++ function)
// This will be called from splat_binding.cpp
void launch_gaussian_splat(
    float* img,
    const float* mu,
    const float* inv_cov,
    const float* clr,
    const float* alpha,
    int W,
    int H,
    int N,
    int nsigma
);