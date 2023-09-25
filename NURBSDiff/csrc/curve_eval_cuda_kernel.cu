#include <torch/extension.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>


#include <vector>



namespace {

__device__ __forceinline__ int find_span(int n, int p, float u, float* U) 
{

      double eps = 1.0e-5;
      if (fabs(u - U[n+1]) < eps)
            return n - 1; 
      int low  = p;
      int high = n+1; 
      int mid = (low + high)/2;
      while (u < U[mid]-eps || u >= U[mid+1]-eps){
            if (u < U[mid]-eps)
                  high = mid;
            else
                  low = mid;
            mid = (low + high)/2;
      }
      return mid;
}


__device__ __forceinline__ void basis_funs(int uspan_i, float u, int p, float* U, float* N, unsigned int i)
{
  float *left  = new float [p+1];
  float *right = new float [p+1];
  float saved, temp;
  int col = p + 1;
  N[i*col] = 1.0;
  for (int j=1; j<=p; j++){
    left[j] = u-U[uspan_i+1-j];
    right[j] = U[uspan_i+j]-u;
    saved = 0.0;
    for(int r = 0; r < j; r++){

      temp = N[i*col  + r]/(right[r+1] + left[j-r]);
      N[i*col+r] = saved + right[r+1]*temp;
      saved = left[j-r]*temp;
    }

    N[i*col+j] = saved;
}

}




__global__ void curve_cuda_pre_compute_basis_kernel(
    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> uspan,
    // torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> Nu,
    torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> u,
    float* U_ptr,
    float* Nu_ptr,
    int m, 
    int p, 
    int out_dim, 
    int _dimension,
    int u_size) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    

    if(i < u_size ){
      
      
      uspan[i] = find_span(m, p, u[i], U_ptr);
      // 
      

      basis_funs(uspan[i],u[i],p,U_ptr,Nu_ptr,i);

      
    }
  }


__global__ void curve_cuda_forward_kernel(
  torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> ctrl_pts,
  torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> uspan,
  torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> Nu,
  torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> u,
  torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> curves,
  // int* uspan,
  // float* Nu,
  int m,
  int p,
  int _dimension,
  unsigned int ctrl_pts_size,
  unsigned int u_size){

  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  // unsigned int l = blockIdx.z * blockDim.z + threadIdx.z;cd t
  // std::printf("Hello from k %d, i %d, l %d\n", k,i,l);


  if(k < ctrl_pts_size)
  { 
    if (i < u_size)
    { 
      for(int l = 0; l <= _dimension; l++)
      { 
        for (int j = 0; j<=p; j++)
        {
          curves[k][i][l] +=  Nu[i][j]*ctrl_pts[k][uspan[i]-p + j][l];
        }
      }
    }
  }
 }




 __global__ void curve_cuda_backward_kernel(
  torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> grad_ctrl_pts,
  torch::PackedTensorAccessor<float,3,torch::RestrictPtrTraits,size_t> grad_output,
  torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> uspan,
  torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> Nu,
  torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> u,
  int m, 
  int p,  
  int _dimension,
  unsigned int curves_size,
  unsigned u_size){

  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  // unsigned int l = blockIdx.z * blockDim.z + threadIdx.z;cd t
  // std::printf("Hello from k %d, i %d, l %d\n", k,i,l);


  if(k < curves_size)
  { 
    if (i < u_size)
    { 
      for(int l = 0; l <= _dimension; l++)
      { 
        for (int j = 0; j <= p; j++)
        {
          grad_ctrl_pts[k][uspan[i]-p+j][l] += Nu[i][j]*grad_output[k][i][l];
        }
      }
    }
  }
 }
   




}






std::vector<torch::Tensor> curve_cuda_pre_compute_basis(
  torch::Tensor u,
  torch::Tensor U,
  // torch::Tensor uspan,
  // torch::Tensor Nu,
  int m,
  int p,
  int out_dim,
  int _dimension){

  float* U_ptr = (float*)U.data_ptr();
  auto options1 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0).requires_grad(false);
  auto options2 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(true);
  
  // auto device = torch::device(torch::kCUDA, 1);
  auto uspan = torch::zeros(u.size(0), options1);
  auto Nu = torch::zeros({u.size(0), p + 1}, options2);
  float* Nu_ptr = (float*)Nu.data_ptr();

  int u_size = u.size(0);

  const dim3 block(1, 1, 1);
  const dim3 grid(u_size+1, 1, 1);

  // AT_DISPATCH_FLOATING_TYPES(u.type(), "curve_cuda_pre_compute", ([&] {
    curve_cuda_pre_compute_basis_kernel<<<grid, block>>>(
        uspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
        // Nu.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
        u.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
        U_ptr,
        Nu_ptr,
        m, 
        p, 
        out_dim, 
        _dimension,
        u_size);
  // }));
    return {uspan, Nu};
  
  }


torch::Tensor curve_cuda_forward(
  torch::Tensor ctrl_pts,
  torch::Tensor uspan,
  torch::Tensor Nu,
  torch::Tensor u,
  int m,
  int p,
  int _dimension){
  
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(true);

  // float* Nu_ptr = (float*)Nu.data_ptr();
  // int* uspan_ptr = (int*)uspan.data_ptr();
  auto curves = torch::zeros({ctrl_pts.size(0),u.size(0), _dimension+1}, options);
  unsigned int ctrl_pts_size = ctrl_pts.size(0);
  unsigned int u_size = u.size(0);

  const dim3 block(16, 16, 1);
  const dim3 grid((ctrl_pts_size)/16+1, (u_size)/16+1, 1);


  curve_cuda_forward_kernel<<<grid, block>>>(
    ctrl_pts.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
    uspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
    Nu.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
    u.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
    curves.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
    m, 
    p,  
    _dimension,
    ctrl_pts_size,
    u_size);


    return curves;

  }


std::vector<torch::Tensor> curve_cuda_backward(
  torch::Tensor grad_output,
  torch::Tensor ctrl_pts,
  torch::Tensor uspan,
  torch::Tensor Nu,
  torch::Tensor u,
  int m,
  int p,
  int _dimension)
  {

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(true);
  auto grad_ctrl_pts = torch::zeros({ctrl_pts.size(0),ctrl_pts.size(1), _dimension+1}, options);
  unsigned int curves_size = ctrl_pts.size(0);
  unsigned int u_size = u.size(0);


  const dim3 block(16, 16, 1);
  const dim3 grid((curves_size)/16+1, (u_size)/16+1, 1);


  curve_cuda_backward_kernel<<<grid, block>>>(
    grad_ctrl_pts.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
    grad_output.packed_accessor<float,3,torch::RestrictPtrTraits,size_t>(),
    uspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
    Nu.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
    u.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
    m, 
    p,  
    _dimension,
    curves_size,
    u_size);

  return {grad_ctrl_pts};


  }