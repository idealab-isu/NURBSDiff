#include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>

std::vector<torch::Tensor> surf_cuda_pre_compute_basis(torch::Tensor u,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor V,
    int m,
    int n, 
    int p,
    int q,
    int out_dim,
    int _dimension);


std::vector<torch::Tensor> surf_cuda_forward(
    torch::Tensor ctrl_pts,
    torch::Tensor u_uv,
    torch::Tensor v_uv,
    torch::Tensor U,
    torch::Tensor V,
    int m,
    int n,
    int p,
    int q,
    int _dimension);


std::vector<torch::Tensor> surf_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts,
    torch::Tensor uspan_uv,
    torch::Tensor vspan_uv,
    torch::Tensor Nu_uv,
    torch::Tensor Nv_uv,
    torch::Tensor u_uv,
    torch::Tensor v_uv,
    int m,
    int n,
    int p,
    int q,
    int _dimension);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> surf_pre_compute_basis(torch::Tensor u,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor V,
    int m,
    int n, 
    int p,
    int q,
    int out_dim,
    int _dimension)

{   
    CHECK_INPUT(u);
    CHECK_INPUT(v);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    std::cout<<"precompute cpp";

return surf_cuda_pre_compute_basis(u,v,U,V,m,n,p,q,out_dim,_dimension);
}


std::vector<torch::Tensor> surf_forward(
    torch::Tensor ctrl_pts,
    torch::Tensor u_uv,
    torch::Tensor v_uv,
    torch::Tensor U,
    torch::Tensor V,
    int m,
    int n,
    int p,
    int q,
    int _dimension
)

{   
    CHECK_INPUT(u_uv);
    CHECK_INPUT(v_uv);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    std::cout<<"precompute cpp";

return surf_cuda_forward(ctrl_pts,u_uv,v_uv,U,V,m,n,p,q,_dimension);

}


// std::vector<torch::Tensor>surf_forward(
//     torch::Tensor ctrl_pts,
//     torch::Tensor u_uv,
//     torch::Tensor v_uv,
//     torch::Tensor U,
//     torch::Tensor V,
//     int m,
//     int n,
//     int p,
//     int q,
//     int _dimension)
//     {

//     std::cout<<"surf_forward";
//     CHECK_INPUT(ctrl_pts);
//     CHECK_INPUT(u_uv);
//     CHECK_INPUT(v_uv);
//     CHECK_INPUT(U);
//     CHECK_INPUT(V);

//     return surf_cuda_forward(ctrl_pts,u_uv,v_uv,U,V,m,n,p,q,_dimension);

//     }

std::vector<torch::Tensor>surf_backward(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts,
    torch::Tensor uspan_uv,
    torch::Tensor vspan_uv,
    torch::Tensor Nu_uv,
    torch::Tensor Nv_uv,
    torch::Tensor u_uv,
    torch::Tensor v_uv,
    int m,
    int n,
    int p,
    int q,
    int _dimension)
    {
        
    CHECK_INPUT(grad_output);
    CHECK_INPUT(ctrl_pts);
    CHECK_INPUT(uspan_uv);
    CHECK_INPUT(vspan_uv);
    CHECK_INPUT(Nu_uv);
    CHECK_INPUT(Nv_uv);
    CHECK_INPUT(u_uv);
    CHECK_INPUT(v_uv);


    return surf_cuda_backward(grad_output,ctrl_pts,uspan_uv,vspan_uv,Nu_uv,Nv_uv,u_uv,v_uv,m,n,p,q,_dimension);

    }



    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
    {
    m.def("pre_compute_basis", &surf_pre_compute_basis, "Pre-Compute Basis");
    m.def("forward", &surf_forward, "Forward function for surface eval");
    m.def("backward",&surf_backward,"Backward function for surface eval");
    }
