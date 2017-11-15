// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_SOLVER_KERNELS_H
#define JAMS_CUDA_SOLVER_KERNELS_H
#include <cufft.h>

#include "cuda_defs.h"
#include "jams/containers/sparsematrix.h"
#include "jblib/containers/sparsematrix.h"

__global__ void cuda_realspace_to_kspace_mapping(const CudaFastFloat * real_data, const int * r_to_k_mapping, const int num_spins, const int nkx, const int nky, const int nkz, CudaFastFloat * kspace_data);

__global__ void cuda_kspace_to_realspace_mapping(const CudaFastFloat * kspace_data, const int * r_to_k_mapping, const int num_spins, const int nkx, const int nky, const int nkz, CudaFastFloat * real_data);

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

#endif  // JAMS_CUDA_SOLVER_KERNELS_H
