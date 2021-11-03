#include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include "utils.h"


std::vector<torch::Tensor> curve_pre_compute_basis(torch::Tensor u,
    torch::Tensor U,
    int m,
    int p,
    int out_dim,
    int _dimension){

  std::vector<int> uspan;
  std::vector<torch::Tensor> Nu;

  float* U_ptr = (float*)U.data_ptr();
  auto options = torch::TensorOptions().dtype(torch::kInt);


  for (int i = 0; i<u.size(0); i++)
  {
    auto uspan_u = find_span(m, p, u[i].item<float>(), U_ptr);
    auto Nu_tensor = torch::zeros({p+1});
    basis_funs(uspan_u, u[i].item<float>(), p, U_ptr, (float*)Nu_tensor.data_ptr());
    uspan.push_back(uspan_u);
    Nu.push_back(Nu_tensor);
  }

  int *uspan_ptr = &uspan[0];
  auto uspan_out = torch::from_blob(uspan_ptr, torch::IntList(u.size(0)), options).clone();
  auto Nu_out = torch::stack(Nu);
  return {uspan_out,
          Nu_out};
}


torch::Tensor curve_forward(
    torch::Tensor ctrl_pts,
    torch::Tensor uspan,
    torch::Tensor Nu,
    torch::Tensor u,
    int m,
    int p,
    int _dimension) {
  // This is for a batch of control points as input and predicting a batch of curves
  auto curve = torch::zeros({ctrl_pts.size(0), u.size(0), _dimension+1}, torch::requires_grad());
  for (int k = 0; k<ctrl_pts.size(0); k++)
  {
    for (int i = 0; i<u.size(0); i++)
    {
      auto Cw = torch::zeros({_dimension+1});
      for (int j = 0; j<=p; j++)
      {
        Cw = Cw + Nu[i][j].item<float>()*ctrl_pts[k][uspan[i].item<int>() - p + j];
      }
      curve[k][i] = Cw;
    }
  }
  return curve;
}


std::vector<torch::Tensor> curve_backward(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts,
    torch::Tensor uspan,
    torch::Tensor Nu,
    torch::Tensor u,
    int m,
    int p,
    int _dimension) {

  auto grad_ctrl_pts = torch::zeros_like(ctrl_pts);

  // std::cout << uspan.size(0)<< Nu.size(1) << std::endl;
  for (int k=0; k<grad_output.size(0); k++)
  {
    for (int i=0; i<u.size(0); i++)
    {
      auto grad_cw = grad_output[k];
      auto grad_ctrl_pts_i = torch::zeros_like(ctrl_pts[k]);
      for (int j = 0; j<=p; j++)
      {
        // std::cout << k << " "<< j << " " << i << " " << uspan[i].item<int>() - p + j << Nu[i][j].item<float>() << grad_cw[i]<< std::endl;
        grad_ctrl_pts_i[uspan[i].item<int>() - p + j] = grad_ctrl_pts_i[uspan[i].item<int>() - p + j] + Nu[i][j].item<float>()*grad_cw[i];
      }
      grad_ctrl_pts[k] = (grad_ctrl_pts[k] + grad_ctrl_pts_i);
    }
  }
  

  return {grad_ctrl_pts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pre_compute_basis", &curve_pre_compute_basis, "Pre-Compute Basis");
  m.def("forward", &curve_forward, "forward func for curve eval");
  m.def("backward", &curve_backward, "forward func for curve eval");
}
