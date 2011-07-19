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
      int colUp  = row+dia_offsets[n];
      int colLow = row-dia_offsets[n];
      float valUp  = dia_values[nrows*n+row];
      float valLow = dia_values[nrows*n+colLow];

      if(colLow >=row && colLow < ncols)
        dot += valLow*x[colLow];

      if(colUp >=0 && colUp < row)
        dot += valUp*x[colUp];
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
          int idxUp  = row + pitch * base;
  
          for(int n = 0; n < chunk_size; n++)
          {
              const int colUp  = row + offsets[n];
              const int colLow = row - offsets[n];

              if(colLow >= row && colLow < ncols) {
                const float A_ij = dia_values[pitch*(base+n)+colLow];
                sum += A_ij * tex1Dfetch(tex_x_float,colLow);
              }
              if(colUp >= 0 && colUp < row) {
                const float A_ij = dia_values[idxUp];
                sum += A_ij * tex1Dfetch(tex_x_float,colUp);
              }

              idxUp += pitch;
          }
  
          y[row] = sum;
      }

      // wait until all threads are done reading offsets 
      __syncthreads();
  }
}
