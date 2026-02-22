#include <torch/extension.h>
#include "splat_kernel.cuh"

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

    auto _mu = mu.contiguous();
    auto _inv_cov = inv_cov.contiguous();
    auto _clr = clr.contiguous();
    auto _alpha = alpha.contiguous();

    auto img = torch::zeros(
        {H, W, 3},
        torch::dtype(torch::kFloat32).device(mu.device())
    );

    launch_gaussian_splat(
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gaussian_splat", &gaussian_splat);
}