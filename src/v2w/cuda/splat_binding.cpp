#include <torch/extension.h>

#define THREADS 256

// Wrapper
torch::Tensor gaussian_splat(
    torch::Tensor mu,
    torch::Tensor inv_cov,
    torch::Tensor clr,
    torch::Tensor alpha,
    int H,
    int W,
    int nsigma
)
{
    int N = mu.size(0);

    mu = mu.contiguous();
    inv_cov = inv_cov.contiguous();
    clr = clr.contiguous();
    alpha = alpha.contiguous();

    auto img = torch::zeros({H, W, 3},
        torch::dtype(torch::kFloat32).device(mu.device()));


    int blocks = (N + THREADS - 1) / THREADS;

    gaussian_splat_kernel<<<blocks, THREADS>>>(
        img.data_ptr<float>(),
        mu.data_ptr<float>(),
        inv_cov.data_ptr<float>(),
        clr.data_ptr<float>(),
        alpha.data_ptr<float>(),
        W,
        H,
        N,
        nsigma
    );

    return img;
}


// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gaussian_splat", &gaussian_splat);
}
