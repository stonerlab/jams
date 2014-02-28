// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_SPARSE_H
#define JAMS_CUDA_SPARSE_H

#include "core/cuda_defs.h"
#include "core/sparsematrix.h"
#include "jblib/containers/sparsematrix.h"

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
