// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/cuda_defs.h"
#include "core/cuda_sparsematrix.h"

#include <thrust/extrema.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "core/globals.h"

#include "core/sparsematrix.h"
#include "core/sparsematrix4d.h"

#include "jblib/containers/sparsematrix.h"

__global__ void cuda_anisotropy_kernel
(
  const int num_spins,
  const CudaFastFloat * dev_d2z_,
  const CudaFastFloat * dev_d4z_,
  const CudaFastFloat * dev_d6z_,
  const CudaFastFloat * dev_sf_,
  CudaFastFloat * dev_h_
) {

  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx < num_spins) {
    const float sz = dev_sf_[3*idx+2];
    dev_h_[3*idx+2] += dev_d2z_[idx]*3.0*sz + dev_d4z_[idx]*(17.5*sz*sz*sz-7.5*sz) + dev_d6z_[idx]*(86.625*sz*sz*sz*sz*sz - 78.75*sz*sz*sz + 13.125*sz);
  }
}

__global__ void spmv_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const size_t pitch,
 const CudaFastFloat alpha,
 const CudaFastFloat beta,
 const int * dia_offsets,
 const CudaFastFloat * dia_values,
 const CudaFastFloat * x,
 CudaFastFloat * y)
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

  if(ndiag == 0){
      for(int row = thread_id; row < nrows; row += grid_size){
        y[row] = 0.0;
      }
  } else {
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
            CudaFastFloat sum;
            if(base == 0){
              // NOTE: floating point comparison avoids reading h_dev[] for
              // special case
              if(beta == 0.0){
                sum=0.0;
              }else{
                sum = beta*y[row];
              }
            } else {
              sum = y[row];
            }

            // index into values array
            int idxUp  = row + pitch * base;

            for(int n = 0; n < chunk_size; n++)
            {
                const int colUp  = row + offsets[n];
                const int colLow = row - offsets[n];

                if(colLow >= row && colLow < ncols) {
                  const CudaFastFloat A_ij = alpha*dia_values[pitch*(base+n)+colLow];
                  sum += A_ij*x[colLow];
                }
                if(colUp >= 0 && colUp < row) {
                  const CudaFastFloat A_ij = alpha*dia_values[idxUp];
                  sum += A_ij*x[colUp];
                }
                idxUp += pitch;
            }

            y[row] = sum;
        }

        // wait until all threads are done reading offsets
        __syncthreads();
    }
  }
}
