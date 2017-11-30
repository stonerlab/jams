// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_SPARSE_H
#define JAMS_CUDA_SPARSE_H
#include <cufft.h>

#include "cuda_defs.h"
#include "jams/containers/sparsematrix.h"
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
  const double * dev_d2z_,
  const double * dev_d4z_,
  const double * dev_d6z_,
  const double * dev_sf_,
  double * dev_h_
);

__global__ void spmv_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const size_t pitch,
 const double alpha,
 const double beta,
 const int * dia_offsets,
 const double * dia_values,
 const double * x,
 double * y);

#endif
