#include <torch/extension.h>
#include <vector>


std::vector<at::Tensor> curve_pre_compute_basis(at::Tensor u,
    at::Tensor U,
    int m,
    int p,
    int out_dim,
    int _dimension);

at::Tensor curve_forward(
    at::Tensor ctrl_pts,
    at::Tensor uspan,
    at::Tensor Nu,
    at::Tensor u,
    int m,
    int p,
    int _dimension);

std::vector<at::Tensor> curve_backward(
    at::Tensor grad_output,
    at::Tensor ctrl_pts,
    at::Tensor uspan,
    at::Tensor Nu,
    at::Tensor u,
    int m,
    int p,
    int _dimension);