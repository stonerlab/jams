// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_ARRAY_KERNELS_H
#define JAMS_CUDA_ARRAY_KERNELS_H

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <jblib/containers/array.h>

#include "jams/cuda/cuda_defs.h"

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

template <typename T, int N>
inline void cuda_copy_array_to_device_pointer(const jblib::Array<T, N> &array, T **dev_ptr) {
  assert(array.is_allocated());
  size_t size_in_bytes = array.elements() * sizeof(T);
  cuda_api_error_check(cudaMalloc((void**)dev_ptr, size_in_bytes));
  cuda_api_error_check(cudaMemcpy(*dev_ptr, array.data(), size_in_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
inline T cuda_reduce_array(T* dev_ptr, const size_t size) {
  return thrust::reduce(thrust::device_ptr<T>(dev_ptr), thrust::device_ptr<T>(dev_ptr) + size);
}
#endif  // JAMS_CUDA_ARRAY_KERNELS_H
