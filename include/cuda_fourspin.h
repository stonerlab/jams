#include <thrust/extrema.h>

#define CSR_4D_BLOCK_SIZE 64

__global__ void fourspin_scalar_csr_kernel
(const int num_rows,
 const int nspins,
 const float alpha,
 const float beta,
 const int * pointers,
 const int * coords,
 const float * val,
 float * h_dev)
{

  const int thread_id = CSR_4D_BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = CSR_4D_BLOCK_SIZE * gridDim.x;

  for(int row = thread_id; row < num_rows; row += grid_size)
  {
    const int row_start = pointers[row];
    const int row_end   = pointers[row+1];

    float sum[3] = {0.0, 0.0, 0.0};

    for(int jj = row_start; jj < row_end; ++jj){
              
      const float A_ijkl = alpha*val[jj];

      const int jidx = coords[jj+0];
      const int kidx = coords[jj+1];
      const int lidx = coords[jj+2];
       
      float sk[3], sl[3];
      
      #pragma unroll
      for(int i=0; i<3; ++i){
        sk[i] = tex1Dfetch(tex_x_float,3*nspins*kidx+i);
      }
      #pragma unroll
      for(int i=0; i<3; ++i){
        sl[i] = tex1Dfetch(tex_x_float,3*nspins*lidx+i);
      }

      float k_dot_l = sk[0]*sl[0] + sk[1]*sl[1] + sk[2]*sl[2];

      #pragma unroll
      for(int i=0; i<3; ++i){
        sum[i] += A_ijkl * tex1Dfetch(tex_x_float,3*nspins*jidx+i)*k_dot_l;
      }
    }

    if(beta == 0.0){ // NOTE: floating point comparison
      #pragma unroll
      for(int i=0; i<3; ++i){
        h_dev[3*row+i] = sum[i];
      }
    }else{
      #pragma unroll
      for(int i=0; i<3; ++i){
        h_dev[3*row+i] = beta*h_dev[3*row+i] + sum[i];
      }
    }
  }
}
