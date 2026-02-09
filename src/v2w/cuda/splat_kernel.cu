#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 256


// Gaussian function
__device__ inline float gaussian_2d(
    float dx,
    float dy,
    const float* inv_cov   // inverse covariance (4)
)
{
    // d^T Σ⁻¹ d
    float q =
        dx*dx*inv_cov[0] +
        dx*dy*inv_cov[1] +
        dy*dx*inv_cov[2] +
        dy*dy*inv_cov[3];

    return __expf(-0.5f * q);
}


// Kernel
__global__ void gaussian_splat_kernel(
    float* img,        // [H*W*3]
    const float* mu,     // [N,2]
    const float* inv_cov,    // [N,2,2]  (inverse cov)
    const float* clr,  // [N,3]
    const float* alpha,  // [N]
    int W,
    int H,
    int N,
    int nsigma
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;


    // Load gaussian parameters
    float mx = mu[2*i + 0];
    float my = mu[2*i + 1];

    const float* inv_cov_i = inv_cov + 4*i;

    float r = clr[3*i + 0];
    float g = clr[3*i + 1];
    float b = clr[3*i + 2];

    float a = alpha[i];


    // Bounding box
    int xmin = max((int)(mx - nsigma), 0);
    int xmax = min((int)(mx + nsigma), W-1);

    int ymin = max((int)(my - nsigma), 0);
    int ymax = min((int)(my + nsigma), H-1);


    // Rasterize gaussian
    for(int y = ymin; y <= ymax; ++y)
    {
        for(int x = xmin; x <= xmax; ++x)
        {
            float dx = x - mx;
            float dy = y - my;

            float w = a * gaussian_2d(dx, dy, inv_cov_i);

            int idx = (y * W + x) * 3;

            // atomic accumulation (important!)
            atomicAdd(&img[idx+0], w*r);
            atomicAdd(&img[idx+1], w*g);
            atomicAdd(&img[idx+2], w*b);
        }
    }
}



