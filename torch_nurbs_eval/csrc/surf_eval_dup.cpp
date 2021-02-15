#include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <utils.h>


std::vector<at::Tensor> pre_compute_basis(at::Tensor u,
    at::Tensor v,
    at::Tensor U,
    torch::Tensor V,
    int m,
    int n, 
    int p,
    int q,
    int out_dim,
    int _dimension){

  std::vector<at::Tensor> uspan_uv;
  std::vector<at::Tensor> vspan_uv;
  std::vector<at::Tensor> Nu_uv;
  std::vector<at::Tensor> Nv_uv;

  float* U_ptr = (float*)U.data_ptr();
  float* V_ptr = (float*)V.data_ptr();
  auto options = torch::TensorOptions().dtype(at::kInt);


  for (int i = 0; i<u.size(0); i++)
  {
    std::vector<int> uspan_v;
    std::vector<int> vspan_v;
    std::vector<at::Tensor> Nu_v;
    std::vector<at::Tensor> Nv_v;
    
    for (auto j=0; j<v.size(0); j++)
    {
      auto uspan = find_span(n, p, u[i].item<float>(), U_ptr);
      auto vspan = find_span(m, q, v[i].item<float>(), V_ptr);
      auto Nu_tensor = torch::zeros({p+1});
      auto Nv_tensor = torch::zeros({q+1});
      basis_funs(uspan, u[i].item<float>(), p, U_ptr, (float*)Nu_tensor.data_ptr());
      basis_funs(vspan, v[i].item<float>(), q, V_ptr, (float*)Nv_tensor.data_ptr());
      uspan_v.push_back(uspan);
      vspan_v.push_back(vspan);
      Nu_v.push_back(Nu_tensor);
      Nv_v.push_back(Nv_tensor);
    }
    int *uspan_v_ptr = &uspan_v[0];
    int *vspan_v_ptr = &vspan_v[0];
    auto uspan_v_tensor = torch::from_blob(uspan_v_ptr, at::IntList(v.size(0)), options);
    auto vspan_v_tensor = torch::from_blob(vspan_v_ptr, at::IntList(v.size(0)), options);
    auto Nu_v_tensor = torch::stack(Nu_v);
    auto Nv_v_tensor = torch::stack(Nv_v);
    uspan_uv.push_back(uspan_v_tensor);
    vspan_uv.push_back(vspan_v_tensor);
    Nu_uv.push_back(Nu_v_tensor);
    Nv_uv.push_back(Nv_v_tensor);
  }

  auto uspan_uv_tensor = torch::stack(uspan_uv);
  auto vspan_uv_tensor = torch::stack(vspan_uv);
  auto Nu_uv_tensor = torch::stack(Nu_uv);
  auto Nv_uv_tensor = torch::stack(Nv_uv);
  return {uspan_uv_tensor,
          vspan_uv_tensor,
          Nu_uv_tensor,
          Nv_uv_tensor};
}


at::Tensor nurbs_forward(
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
    int _dimension) {
  auto surface = torch::zeros({ctrl_pts.size(0), u_uv.size(0), v_uv.size(0), _dimension}, torch::requires_grad());
  for (int k = 0; k<ctrl_pts.size(0); k++)
  {
    for (int j = 0; j<v_uv.size(0); j++)
    {
      for (int i = 0; i<u_uv.size(0); i++)
      {
        auto temp = torch::zeros({q+1, _dimension+1});
        for (int l = 0; l<=q; l++)
        {
          for (int r = 0; r<=p; r++)
          {
            // std::cout << uspan_uv << std::endl;
            // std::cout << uspan_uv[i][j] << vspan_uv[i][j] << std::endl;
            // std::cout << temp << l << " " << r << q << p << std::endl;
            temp[l] = temp[l] + Nu_uv[i][j][r].item<float>()*ctrl_pts[k][uspan_uv[i][j].item<int>() - p - r][vspan_uv[i][j].item<int>() - q - l];
            // std::cout << temp << l << " " << r << std::endl;
            // std::cout << Nu_uv[i][j][r].item<float>() << ctrl_pts[k][uspan_uv[i][j].item<int>() - p - r][vspan_uv[i][j].item<int>() - q - l] << std::endl;
            // std::cout << "debug 1" << std::endl;
          }
        }
        // std::cout << "debug 2" << std::endl;
        auto Sw = torch::zeros({_dimension+1});
        for (int l=0; l<=q; l++)
          Sw = Sw + Nv_uv[i][j][l]*temp[l];
        // std::cout << "debug 3" << std::endl;
        surface[k][i][j] = Sw.slice(0,0,3)*Sw.slice(0,3,4);
        // std::cout << "debug 4" << std::endl;
      }
    }
  }
  return surface;
}

std::vector<at::Tensor> nurbs_backward(
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
    int _dimension) {

  auto grad_ctrl_pts = torch::zeros_like(ctrl_pts);

  for (int k=0; k<grad_output.size(0); k++)
  {
    for (int j=0; j<v_uv.size(0); j++)
    {
      for (int i=0; i<u_uv.size(0); i++)
      {
        auto grad_sw = torch::zeros({_dimension+1});
        grad_sw[0] = grad_output[k][i][j][0];
        grad_sw[1] = grad_output[k][i][j][1];
        grad_sw[2] = grad_output[k][i][j][2];
        grad_sw[3] = grad_output[k][i][j][0]/ctrl_pts[k][i][j][0] + grad_output[k][i][j][1]/ctrl_pts[k][i][j][1] + grad_output[k][i][j][2]/ctrl_pts[k][i][j][2];

        auto grad_temp = torch::zeros({p+1,_dimension+1});
        for (int l =0; l<=q; l++)
        {
          grad_temp[l] = grad_temp[l] + grad_sw*Nv_uv[i][j][l];
        }
        for (int l = 0; l<=q; l++)
        {
          for (int r = 0; r<=p; r++)
          {
            grad_ctrl_pts[k][uspan_uv[i][j].item<int>() - p - r][vspan_uv[i][j].item<int>() - q - l] = grad_ctrl_pts[k][uspan_uv[i][j].item<int>() - p - r][vspan_uv[i][j].item<int>() - q - l] + Nu_uv[i][j][r].item<float>()*grad_temp[l];
          }
        }

      }
    }
  }

  return {grad_ctrl_pts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nurbs_forward, "forward");
  m.def("backward", &nurbs_backward, "backward");
  m.def("pre_compute_basis", &pre_compute_basis, "Pre-Compute Basis");
}
