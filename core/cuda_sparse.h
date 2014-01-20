// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_SPARSE_H
#define JAMS_CUDA_SPARSE_H

#include "core/cuda_sparse_types.h"
#include "core/sparsematrix.h"

#include "jblib/containers/sparsematrix.h"

// block size for GPU, 64 appears to be most efficient for current kernel
#define BLOCKSIZE 64
#define DIA_BLOCK_SIZE 256
#define CSR_4D_BLOCK_SIZE 64

// extern texture<float, 1> tex_x_float;

void allocate_transfer_dia(SparseMatrix<float> &Jij, devDIA &Jij_dev);

void free_dia(devDIA &Jij_dev);

void allocate_transfer_csr_4d(jblib::Sparsematrix<float, 4> &Jij, devCSR &
    Jij_dev);

void free_csr_4d(devCSR &Jij_dev);

__global__ void bruteforce_dipole_interaction_kernel
(
 const float alpha,
 const float beta,
 const float *sf_dev,
 const float *mat_dev,
 float *h_dev,
 const float *r_dev,
 const float *r_max_dev,
 const bool *pbc_dev,
 const int num_spins
);

__global__ void bilinear_scalar_interaction_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const float alpha,
 const float beta,
 const int * dia_offsets,
 const float * dia_values,
 const float * sf_dev,
 float * h_dev);

__global__ void biquadratic_scalar_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const float alpha,
 const float beta,
 const int * dia_offsets,
 const float * dia_values,
 const float * sf_dev,
 float * h_dev);

__global__ void spmv_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const float alpha,
 const float beta,
 const int * dia_offsets,
 const float * dia_values,
 const float * x,
 float * y);

__global__ void fourspin_scalar_interaction_csr_kernel
(const int num_rows,
 const int num_spins,
 const float alpha,
 const float beta,
 const int * pointers,
 const int * coords,
 const float * val,
 const float * sf_dev,
 float * h_dev);

#endif
