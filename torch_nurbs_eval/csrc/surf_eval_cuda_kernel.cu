#include <torch/extension.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>


#include <vector>


namespace {

__device__ __forceinline__ int find_span(int n, int p, float u, float* U)

{

double eps = 1.0e-4;
if (fabs(u-U[n+1]) < eps)
    return n;
int low = p;
int high = n+1;
int mid = (low + high)/2;
while (u < U[mid]-eps || u>=U[mid+1]+eps)
{ if (u < U[mid]-eps)
    high = mid;
    else
    low = mid;
    mid = (low+high)/2;
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
      // temp = N[i*col  + r]/((U[uspan_i + r + 1] - u) + (u - U[uspan_i  + 1 - j + r]));
      // N[i*col+r] = saved + (U[uspan_i + r + 1] - u) * temp;
      N[i*col+r] = saved + right[r+1]*temp;
      // saved = (u - U[uspan_i + 1 - j + r]) * temp;
      saved = left[j-r]*temp;
    }
    N[i*col+j] = saved;
  }

}




__global__ void surf_cuda_pre_compute_basis_kernel(
    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> uspan,
    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> vspan,
    // torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> Nu,
    torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> u,
    torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> v,
    float* U_ptr,
    float* V_ptr,
    float* Nu_ptr,
    float* Nv_ptr,
    int m, 
    int n,
    int p, 
    int q,
    int out_dim, 
    int _dimension,
    int u_size,
    int v_size) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    

    if(i < u_size ){


    uspan[i]= find_span(m, p, u[i], U_ptr);
    basis_funs(uspan[i],u[i],p,U_ptr,Nu_ptr,i);

    }

    if (j < v_size){
      
    vspan[j] = find_span(n, q, v[j], V_ptr);
    basis_funs(vspan[j],v[j],q,V_ptr,Nv_ptr,j);

    }
    

  }


__global__ void surf_cuda_forward_kernel(
  torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> ctrl_pts,
  torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> uspan,
  torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> vspan,
  torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> Nu,
  torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> Nv,
  torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> u,
  torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> v,
  torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> surfaces,
  torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> temp,
  // int* uspan,
  // float* Nu,
  int m,
  int n,
  int p,
  int q,
  int _dimension,
  unsigned int ctrl_pts_size,
  unsigned int u_size,
  unsigned int v_size){

  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.z * blockDim.z + threadIdx.z;
  // std::printf("Hello from k %d, i %d, l %d\n", k,i,l);


  if(k < ctrl_pts_size )
  { if (j < v_size )
    { if (i < u_size )
      { 

        for (int d=0; d<=_dimension; d++)
          {
            for (int l = 0; l<=q; l++)
            {
              for (int r = 0; r <=p ; r++)
              { temp[l][d] = temp[l][d] + Nu[i][r]*ctrl_pts[k][uspan[i]-p+r][vspan[j]-q+l][d];
              
              }
            
            
            surfaces[k][i][j][d] = surfaces[k][i][j][d] + Nv[j][l]*temp[l][d];

            }

          

          }
  
      }

    }
  }
 
}
   
__global__ void surf_cuda_backward_kernel(
  torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_output,
  torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> ctrl_pts,
  torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> uspan,
  torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> vspan,
  torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> Nu,
  torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> Nv,
  torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> u,
  torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> v,
  torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_ctrl_pts,
  torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> grad_temp,
  // int* uspan,
  // float* Nu,
  int m,
  int n,
  int p,
  int q,
  int _dimension,
  unsigned int grad_output_size,
  unsigned int u_size,
  unsigned int v_size){

  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.z * blockDim.z + threadIdx.z;
  // std::printf("Hello from k %d, i %d, l %d\n", k,i,l);


  if(k < grad_output_size )
  { if ( j< v_size )
    { if( i < u_size )
      { 

        for (int d=0; d<=_dimension; d++)
          {
            for (int l = 0; l<=q; l++)
            {   grad_temp[l][d] = Nv[j][l]*grad_output[k][i][j][d];
              for (int r = 0; r <=p ; r++)
              { 
                grad_ctrl_pts[k][ uspan[i] - p + r][ vspan[j] - q + l ][d] = grad_ctrl_pts[k][ uspan[i] - p + r][ vspan[j] - q + l ][d] + Nu[i][r]*grad_temp[l][d];
              
              }
            
            
            

            }

          

          }
  
      }

    }
  }
 
}
   




} //namespace end



std::vector<torch::Tensor> surf_cuda_pre_compute_basis(
    torch::Tensor u,
    torch::Tensor v,
    torch::Tensor U,
    torch::Tensor V,
    int m,
    int n,
    int p,
    int q,
    int out_dim,
    int _dimension){
  
    float* U_ptr = (float*)U.data_ptr();
    float* V_ptr = (float*)V.data_ptr();
    auto options1 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0).requires_grad(false);
    auto options2 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(true);
    
    // auto device = torch::device(torch::kCUDA, 1);
    auto uspan = torch::zeros(u.size(0), options1);
    auto vspan = torch::zeros(v.size(0), options1);
    auto Nu = torch::zeros({(u.size(0))*( p + 1)}, options2);
    auto Nv = torch::zeros({(v.size(0))*(q + 1)}, options2);
    float* Nu_ptr = (float*)Nu.data_ptr();
    float* Nv_ptr = (float*)Nv.data_ptr();
  
    int u_size = u.size(0);
    int v_size = v.size(0);
  
    const dim3 block(32, 32, 1);
    const dim3 grid(u_size/32+1, v_size/32+1, 1);
  
    // AT_DISPATCH_FLOATING_TYPES(u.type(), "curve_cuda_pre_compute", ([&] {
      surf_cuda_pre_compute_basis_kernel<<<grid, block>>>(
          uspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
          vspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
          // Nu.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
          u.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
          v.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
          U_ptr,
          V_ptr,
          Nu_ptr,
          Nv_ptr,
          m, 
          n, 
          p, 
          q, 
          out_dim, 
          _dimension,
          u_size,
          v_size);
    // }));
  
      return {uspan, vspan, Nu, Nv};
    
    }



    

torch::Tensor surf_cuda_forward(
    torch::Tensor ctrl_pts,
    torch::Tensor uspan,
    torch::Tensor vspan,
    torch::Tensor Nu,
    torch::Tensor Nv,
    torch::Tensor u,
    torch::Tensor v,
    int m,
    int n,
    int p,
    int q,
    int _dimension){
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(true);
  
    // float* Nu_ptr = (float*)Nu.data_ptr();
    // int* uspan_ptr = (int*)uspan.data_ptr();
    auto surfaces = torch::zeros({ctrl_pts.size(0),u.size(0), v.size(0), _dimension+1}, options);
    auto temp = torch::zeros({q+1, _dimension+1},options);
    unsigned int ctrl_pts_size = ctrl_pts.size(0);
    unsigned int u_size = u.size(0);
    unsigned int v_size = v.size(0);
  
    const dim3 block(16, 16, 4);
    const dim3 grid((ctrl_pts_size)/16+1, (u_size)/16+1, (v_size)/4+1);
  
  
    surf_cuda_forward_kernel<<<grid, block>>>(
      ctrl_pts.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
      uspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
      vspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
      Nu.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
      Nv.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
      u.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
      v.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
      surfaces.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
      temp.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
      m, 
      n,  
      p,
      q,
      _dimension,
      ctrl_pts_size,
      u_size,
      v_size);
  
  
      return surfaces;
  
    }







    std::vector<torch::Tensor>surf_cuda_backward(
      torch::Tensor grad_output,
      torch::Tensor ctrl_pts,
      torch::Tensor uspan,
      torch::Tensor vspan,
      torch::Tensor Nu,
      torch::Tensor Nv,
      torch::Tensor u,
      torch::Tensor v,
      int m,
      int n,
      int p,
      int q,
      int _dimension){
      
      auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(true);
    
      // float* Nu_ptr = (float*)Nu.data_ptr();
      // int* uspan_ptr = (int*)uspan.data_ptr();
      auto grad_ctrl_pts = torch::zeros({ctrl_pts.size(0),ctrl_pts.size(1), ctrl_pts.size(2), _dimension+1}, options);
      auto grad_temp = torch::zeros({q+1, _dimension+1},options);
      unsigned int grad_output_size = grad_output.size(0);
      unsigned int u_size = u.size(0);
      unsigned int v_size = v.size(0);
    
      const dim3 block(16, 16, 4);
      const dim3 grid((grad_output_size)/16+1, (u_size)/16+1, (v_size)/4+1);
    
    
      surf_cuda_backward_kernel<<<grid, block>>>(
        grad_output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        ctrl_pts.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        uspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
        vspan.packed_accessor<int,1,torch::RestrictPtrTraits,size_t>(),
        Nu.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
        Nv.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
        u.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
        v.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
        grad_ctrl_pts.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        grad_temp.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
        m, 
        n,
        p,
        q,  
        _dimension,
        grad_output_size,
        u_size,
        v_size);
    
    
        return {grad_ctrl_pts};
    
      }