#include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>

std::vector<torch::Tensor> curve_cuda_pre_compute_basis(
    torch::Tensor u,
    torch::Tensor U,
    // torch::Tensor uspan,
    // torch::Tensor Nu,
    int m,
    int p,
    int out_dim,
    int _dimension);


torch::Tensor curve_cuda_forward(
    torch::Tensor ctrl_pts,
    torch::Tensor uspan,
    torch::Tensor Nu,
    torch::Tensor u,
    int m,
    int p,
    int _dimension);


std::vector<torch::Tensor> curve_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts,
    torch::Tensor uspan,
    torch::Tensor Nu,
    torch::Tensor u,
    int m,
    int p,
    int _dimension);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> curve_pre_compute_basis(
    torch::Tensor u,
    torch::Tensor U,
    // torch::Tensor uspan,
    // torch::Tensor Nu,
    int m,
    int p,
    int out_dim,
    int _dimension)

{
CHECK_INPUT(u);
CHECK_INPUT(U);
// CHECK_INPUT(uspan);
// CHECK_INPUT(Nu);

return curve_cuda_pre_compute_basis(u,U,m,p,out_dim,_dimension);

}



torch::Tensor curve_forward(
    torch::Tensor ctrl_pts,
    torch::Tensor uspan,
    torch::Tensor Nu,
    torch::Tensor u,
    int m,
    int p,
    int _dimension)
{
CHECK_INPUT(ctrl_pts);
CHECK_INPUT(uspan);
CHECK_INPUT(Nu);
CHECK_INPUT(u);

return curve_cuda_forward(ctrl_pts,uspan,Nu,u,m,p,_dimension);

}

std::vector<torch::Tensor> curve_backward(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts,
    torch::Tensor uspan,
    torch::Tensor Nu,
    torch::Tensor u,
    int m,
    int p,
    int _dimension)

{
CHECK_INPUT(grad_output);
CHECK_INPUT(ctrl_pts);
CHECK_INPUT(uspan);
CHECK_INPUT(Nu);
CHECK_INPUT(u);
return curve_cuda_backward(grad_output,ctrl_pts,uspan,Nu,u,m,p,_dimension);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pre_compute_basis", &curve_pre_compute_basis, "Pre-Compute Basis");
  m.def("forward", &curve_forward, "Forward func for Curve eval");
  m.def("backward", &curve_backward, "Backward func for Curve eval");
}

