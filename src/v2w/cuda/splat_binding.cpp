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

    torch::Tensor _mu = mu.contiguous();
    torch::Tensor _inv_cov = inv_cov.contiguous();
    torch::Tensor _clr = clr.contiguous();
    torch::Tensor _alpha = alpha.contiguous();

    auto img = torch::zeros({H, W, 3},
        torch::dtype(torch::kFloat32).device(mu.device()));


    int blocks = (N + THREADS - 1) / THREADS;

    gaussian_splat_kernel<<<blocks, THREADS>>>(
        img.data_ptr<float>(),
        _mu.data_ptr<float>(),
        _inv_cov.data_ptr<float>(),
        _clr.data_ptr<float>(),
        _alpha.data_ptr<float>(),
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
