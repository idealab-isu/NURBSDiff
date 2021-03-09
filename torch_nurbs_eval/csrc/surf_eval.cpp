#include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include "utils.h"


std::vector<torch::Tensor> surf_pre_compute_basis(torch::Tensor u,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor V,
    int m,
    int n, 
    int p,
    int q,
    int out_dim,
    int _dimension){


  std::vector<int> uspan;
  std::vector<torch::Tensor> Nu;

  std::vector<int> vspan;
  std::vector<torch::Tensor> Nv;

  // std::vector<torch::Tensor> uspan_uv;
  // std::vector<torch::Tensor> vspan_uv;
  // std::vector<torch::Tensor> Nu_uv;
  // std::vector<torch::Tensor> Nv_uv;

  float* U_ptr = (float*)U.data_ptr();
  float* V_ptr = (float*)V.data_ptr();
  auto options = torch::TensorOptions().dtype(torch::kInt);

  for (int i = 0; i<u.size(0); i++)
  {
    auto uspan_u = find_span(m, p, u[i].item<float>(), U_ptr);
    auto Nu_tensor = torch::zeros({p+1});
    basis_funs(uspan_u, u[i].item<float>(), p, U_ptr, (float*)Nu_tensor.data_ptr());
    uspan.push_back(uspan_u);
    Nu.push_back(Nu_tensor);
  }


  for (int j = 0; j<v.size(0); j++)
  {
    auto vspan_v = find_span(n, q, v[j].item<float>(), V_ptr);
    auto Nv_tensor = torch::zeros({q+1});
    basis_funs(vspan_v, v[j].item<float>(), q, V_ptr, (float*)Nv_tensor.data_ptr());
    vspan.push_back(vspan_v);
    Nv.push_back(Nv_tensor);
  }



  int *uspan_ptr = &uspan[0];
  auto uspan_out = torch::from_blob(uspan_ptr, torch::IntList(u.size(0)), options).clone();
  auto Nu_out = torch::stack(Nu);

  int *vspan_ptr = &vspan[0];
  auto vspan_out = torch::from_blob(vspan_ptr, torch::IntList(v.size(0)), options).clone();
  auto Nv_out = torch::stack(Nv);




  // for (int i = 0; i<u.size(0); i++)
  // {
  //   std::vector<int> uspan_v;
  //   std::vector<int> vspan_v;
  //   std::vector<torch::Tensor> Nu_v;
  //   std::vector<torch::Tensor> Nv_v;


  //   auto Nu_tensor = torch::zeros({p+1});
  //   auto uspan = find_span(n, p, u[i].item<float>(), U_ptr);
  //   basis_funs(uspan, u[i].item<float>(), p, U_ptr, (float*)Nu_tensor.data_ptr());
    
  //   for (auto j=0; j<v.size(0); j++)
  //   {
      
  //     auto vspan = find_span(m, q, v[j].item<float>(), V_ptr);
      
  //     auto Nv_tensor = torch::zeros({q+1});
      
  //     basis_funs(vspan, v[j].item<float>(), q, V_ptr, (float*)Nv_tensor.data_ptr());
  //     uspan_v.push_back(uspan);
  //     vspan_v.push_back(vspan);
  //     Nu_v.push_back(Nu_tensor);
  //     Nv_v.push_back(Nv_tensor);
  //   }
  //   int *uspan_v_ptr = &uspan_v[0];
  //   int *vspan_v_ptr = &vspan_v[0];
  //   auto uspan_v_tensor = torch::from_blob(uspan_v_ptr, torch::IntList(v.size(0)), options).clone();
  //   auto vspan_v_tensor = torch::from_blob(vspan_v_ptr, torch::IntList(v.size(0)), options).clone();
  //   auto Nu_v_tensor = torch::stack(Nu_v);
  //   auto Nv_v_tensor = torch::stack(Nv_v);
  //   uspan_uv.push_back(uspan_v_tensor);
  //   vspan_uv.push_back(vspan_v_tensor);
  //   Nu_uv.push_back(Nu_v_tensor);
  //   Nv_uv.push_back(Nv_v_tensor);
  // }

  // auto uspan_uv_tensor = torch::stack(uspan_uv);
  // auto vspan_uv_tensor = torch::stack(vspan_uv);
  // auto Nu_uv_tensor = torch::stack(Nu_uv);
  // auto Nv_uv_tensor = torch::stack(Nv_uv);



  return {uspan_out,
          vspan_out,
          Nu_out,
          Nv_out};
}


torch::Tensor surf_forward(
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
    int _dimension) {
  auto surface = torch::zeros({ctrl_pts.size(0), u_uv.size(0), v_uv.size(0), _dimension+1}, torch::requires_grad());
  for (int k = 0; k<ctrl_pts.size(0); k++)
  {
    for (int j = 0; j<v_uv.size(0); j++)
    {
      for (int i = 0; i<u_uv.size(0); i++)
      {
        auto temp = torch::zeros({q+1, _dimension+1});
        auto Sw = torch::zeros({_dimension+1});
        for (int l = 0; l<=q; l++)
        {
          for (int r = 0; r<=p; r++)
          {
            temp[l] = temp[l] + Nu_uv[i][r].item<float>()*ctrl_pts[k][uspan_uv[i].item<int>() - p + r][vspan_uv[j].item<int>() - q + l];
          }
          Sw += Nv_uv[j][l].item<float>()*temp[l];
        }
        surface[k][i][j] = Sw;
      }
    }
  }
  return surface;
}

std::vector<torch::Tensor> surf_backward(
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
    int _dimension) {

  auto grad_ctrl_pts = torch::zeros_like(ctrl_pts);

  for (int k=0; k<grad_output.size(0); k++)
  {
    for (int j=0; j<v_uv.size(0); j++)
    {
      for (int i=0; i<u_uv.size(0); i++)
      {
        auto grad_temp = torch::zeros({q+1,_dimension+1});
        for (int l = 0; l<=q; l++)
        {
          grad_temp[l] = Nv_uv[j][l]*grad_output[k][i][j];
          for (int r = 0; r<=p; r++)
          {
            grad_ctrl_pts[k][uspan_uv[i].item<int>() - p + r][vspan_uv[j].item<int>() - q + l] = grad_ctrl_pts[k][uspan_uv[i].item<int>() - p + r][vspan_uv[j].item<int>() - q + l] + Nu_uv[i][r].item<float>()*grad_temp[l];
          }
        }
      }
    }
  }
  return {grad_ctrl_pts};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pre_compute_basis", &surf_pre_compute_basis, "Pre-Compute Basis");
  m.def("forward", &surf_forward, "forward func for curve eval");
  m.def("backward", &surf_backward, "forward func for curve eval");
}
