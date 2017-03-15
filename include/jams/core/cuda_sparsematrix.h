// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_SPARSE_H
#define JAMS_CUDA_SPARSE_H
#include <cufft.h>

#include "jams/core/cuda_defs.h"
#include "jams/core/sparsematrix.h"
#include "jblib/containers/sparsematrix.h"

__global__ void cuda_fft_convolution
(
  const int size,
  const int realsize,
  const cufftDoubleComplex *dev_wq,
  const cufftDoubleComplex *dev_sq,
  cufftDoubleComplex *dev_hq
);

__global__ void cuda_anisotropy_kernel
(
  const int num_spins,
  const CudaFastFloat * dev_d2z_,
  const CudaFastFloat * dev_d4z_,
  const CudaFastFloat * dev_d6z_,
  const CudaFastFloat * dev_sf_,
  CudaFastFloat * dev_h_
);

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
 CudaFastFloat * y);

#endif
