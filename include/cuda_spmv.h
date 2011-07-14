#include <thrust/extrema.h>

#define BLOCK_SIZE 256

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
  __shared__ int offsets[BLOCK_SIZE];

  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = BLOCK_SIZE * gridDim.x;

  for(int base = 0; base < ndiag; base += BLOCK_SIZE)
  {
      // read a chunk of the diagonal offsets into shared memory
      const int chunk_size = thrust::min(int(BLOCK_SIZE), ndiag - base);

      if(threadIdx.x < chunk_size)
          offsets[threadIdx.x] = dia_offsets[base + threadIdx.x];
  
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
                  sum += A_ij * x[col];
              }
      
              idx += pitch;
          }
  
          y[row] = sum;
      }

      // wait until all threads are done reading offsets 
      __syncthreads();
  }
}
