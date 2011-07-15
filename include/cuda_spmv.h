#include <thrust/extrema.h>

#define DIA_BLOCK_SIZE 256

texture<float,1> tex_x_float;

__global__ void spmv_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const int * dia_offsets,
 const float * dia_values,
 const float * x,
 float * y)
{

/*
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if(row < nrows){
    float dot=0;
    for(int n=0;n<ndiag;++n){
      int col = row+dia_offsets[n];
      float val = dia_values[nrows*n+row];

      if(col >=0 && col < ncols)
        dot += val*x[col];
    }
    y[row] = dot;
  }
  */
  __shared__ int offsets[DIA_BLOCK_SIZE];

  const int thread_id = DIA_BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = DIA_BLOCK_SIZE * gridDim.x;

  for(int base = 0; base < ndiag; base += DIA_BLOCK_SIZE)
  {
      // read a chunk of the diagonal offsets into shared memory
      const int chunk_size = thrust::min(int(DIA_BLOCK_SIZE), ndiag - base);

      if(threadIdx.x < chunk_size) {
          offsets[threadIdx.x] = dia_offsets[base + threadIdx.x];
      }
  
      __syncthreads();
 
      // process chunk
      for(int row = thread_id; row < nrows; row += grid_size)
      {
          float sum = (base == 0) ? float(0) : y[row];
  
          // index into values array
          int idx = row + pitch * base;
  
          for(int n = 0; n < chunk_size; n++)
          {
              const int col = row + offsets[n];
      
              if(col >= 0 && col < ncols)
              {
                  const float A_ij = dia_values[idx];
                  //sum += A_ij * x[col];
                  sum += A_ij * tex1Dfetch(tex_x_float,col);
              }
      
              idx += pitch;
          }
  
          y[row] = sum;
      }

      // wait until all threads are done reading offsets 
      __syncthreads();
  }
}
