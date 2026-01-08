// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_ARRAY_KERNELS_H
#define JAMS_CUDA_ARRAY_KERNELS_H

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "jams/cuda/cuda_common.h"

void cuda_array_elementwise_scale(
	const unsigned int n,            // n elements in i index
	const unsigned int m,            // m elements in j index
	const float * alpha,   // scale factors array of length n
	const float   beta,    // uniform scale factor
	float * x,             // input array
	const unsigned int incx,         // input increment
	float * y,             // output array
	const unsigned int incy,         // output increment
	cudaStream_t stream     // cuda stream
);

void cuda_array_elementwise_scale(
    const unsigned int n,            // n elements in i index
    const unsigned int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const unsigned int incx,         // input increment
    double * y,             // output array
    const unsigned int incy,         // output increment
    cudaStream_t stream     // cuda stream
);

void cuda_array_double_to_float(
	const unsigned int n,            // n elements in i index
	const double * in,
	float * out,  
	cudaStream_t stream     // cuda stream
);

void cuda_array_float_to_double(
	const unsigned int n,            // n elements in i index
	const float * in,
	double * out,  
	cudaStream_t stream     // cuda stream
);


// Calculates the dot product of two (n,3) arrays along the second axis
void cuda_array_dot_product(
	unsigned int n,
	const double A,
	const double * x,
	const double * y,
	double * out,
	cudaStream_t stream
	);

__global__ void cuda_array_sum_across(
	unsigned int num_input_arrays,
	unsigned int num_elements,
	double** inputs,
	double* out
	);

double cuda_reduce_array(const double* dev_ptr, const size_t size, cudaStream_t stream = nullptr);

#endif
#endif  // JAMS_CUDA_ARRAY_KERNELS_H
