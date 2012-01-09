#include <thrust/extrema.h>

#define DIA_BLOCK_SIZE 256

__global__ void bilinear_scalar_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const float alpha,
 const float beta,
 const int * dia_offsets,
 const float * dia_values,
 float * sf_dev,
 float * h_dev)
{
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
        const int rowOff = 3*row;
        float sum[3];
        if(base == 0){
          // NOTE: floating point comparison avoids reading h_dev[] for
          // special case
          if(beta == 0.0){
            #pragma unroll
            for(int i=0; i<3; ++i){
              sum[i] = 0.0;
            }
          } else {
            // read initial sum values
            #pragma unroll
            for(int i=0; i<3; ++i){
              sum[i] = beta*h_dev[rowOff+i];
            }
          }
        } else {
          // outside base 0 use existing values
          #pragma unroll
          for(int i=0; i<3; ++i){
            sum[i] = h_dev[rowOff+i];
          }
        }

        // index into values array
        int idxUp  = row + pitch * base;

        for(int n = 0; n < chunk_size; n++)
        {
            const int colUp  = row + offsets[n];
            const int colLow = row - offsets[n];

            if(colLow >= row && colLow < ncols) {
              const int sj = 3*colLow;

              const float A_ij = alpha*dia_values[pitch*(base+n)+colLow];

              #pragma unroll
              for(int i=0; i<3; ++i){
                sum[i] += A_ij * sf_dev[sj+i];
              }
            }
            if(colUp >= 0 && colUp < row) {
              const int sj = 3*colUp;

              const float A_ij = alpha*dia_values[idxUp];
              
              #pragma unroll
              for(int i=0; i<3; ++i){
                sum[i] += A_ij * sf_dev[sj+i];
              }
            }

            idxUp += pitch;
        }

        #pragma unroll
        for(int i=0; i<3; ++i){
          h_dev[rowOff+i] = sum[i];
        }
      }

      // wait until all threads are done reading offsets 
      __syncthreads();
  }

}

__global__ void biquadratic_scalar_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const float alpha,
 const float beta,
 const int * dia_offsets,
 const float * dia_values,
 float * sf_dev,
 float * h_dev)
{
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
        const int rowOff = 3*row;
        float sum[3];
        
        if(base == 0){
          // NOTE: floating point comparison avoids reading h_dev[] for
          // special case
          if(beta == 0.0){
            #pragma unroll
            for(int i=0; i<3; ++i){
              sum[i] = 0.0;
            }
          } else {
            // read initial sum values
            #pragma unroll
            for(int i=0; i<3; ++i){
              sum[i] = beta*h_dev[rowOff+i];
            }
          }
        } else {
          // outside base 0 use existing values
          #pragma unroll
          for(int i=0; i<3; ++i){
            sum[i] = h_dev[rowOff+i];
          }
        }

        // index into values array
        int idxUp  = row + pitch * base;

        for(int n = 0; n < chunk_size; n++)
        {
            const int colUp  = row + offsets[n];
            const int colLow = row - offsets[n];

            if(colLow >= row && colLow < ncols) {
              const int si = rowOff; // 3*row
              const int sj = 3*colLow;

              const float prod = sf_dev[si+0]*sf_dev[sj+0] 
                              + sf_dev[si+1]*sf_dev[sj+1]
                              + sf_dev[si+2]*sf_dev[sj+2];
                
              const float A_ij = alpha*prod*dia_values[pitch*(base+n)+colLow];

              for(int i=0; i<3; ++i){
                sum[i] += A_ij * sf_dev[sj+i];
              }
            }
            if(colUp >= 0 && colUp < row) {
              const int si = rowOff; // 3*row
              const int sj = 3*colUp;

              const float prod = sf_dev[si+0]*sf_dev[sj+0] 
                              + sf_dev[si+1]*sf_dev[sj+1]
                              + sf_dev[si+2]*sf_dev[sj+2];
                
              const float A_ij = alpha*prod*dia_values[idxUp];
              
              for(int i=0; i<3; ++i){
                sum[i] += A_ij * sf_dev[sj+i];
              }
            }

            idxUp += pitch;
        }

        for(int i=0; i<3; ++i){
          h_dev[rowOff+i] = sum[i];
        }
      }

      // wait until all threads are done reading offsets 
      __syncthreads();
  }

}
