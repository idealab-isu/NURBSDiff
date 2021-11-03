#include <torch/extension.h>
#include <vector>


std::vector<at::Tensor> surf_pre_compute_basis(at::Tensor u,
    at::Tensor v,
    at::Tensor U,
    torch::Tensor V,
    int m,
    int n, 
    int p,
    int q,
    int out_dim,
    int _dimension);

at::Tensor surf_forward(
    at::Tensor ctrl_pts,
    at::Tensor uspan_uv,
    at::Tensor vspan_uv,
    at::Tensor Nu_uv,
    at::Tensor Nv_uv,
    at::Tensor u_uv,
    at::Tensor v_uv,
    int m,
    int n,
    int p,
    int q,
    int _dimension);

std::vector<at::Tensor> surf_backward(
    at::Tensor grad_output,
    at::Tensor ctrl_pts,
    at::Tensor uspan_uv,
    at::Tensor vspan_uv,
    at::Tensor Nu_uv,
    at::Tensor Nv_uv,
    at::Tensor u_uv,
    at::Tensor v_uv,
    int m,
    int n,
    int p,
    int q,
    int _dimension);