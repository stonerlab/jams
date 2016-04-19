// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_ARRAY_KERNELS_H
#define JAMS_CUDA_ARRAY_KERNELS_H

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

void cuda_array_elementwise_daxpy(
    const unsigned int n,            // n elements in i index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const unsigned int incx,         // input increment
    double * y,             // output array
    const unsigned int incy,         // output increment
    cudaStream_t stream     // cuda stream
);

void cuda_array_remapping(
    const unsigned int n,   // n elements in array
    const int * map,          // remapping array
    const double * x,
          double * y,
    cudaStream_t stream     // cuda stream
);

void cuda_array_initialize(
    const size_t n,   // n elements in array
    double * x,
    const double value,
    cudaStream_t stream     // cuda stream
);

void cuda_array_elementwise_cos(
    const size_t n,            // n elements in i index
    const double * theta,  // array frequency
    const double   phi,    // constant frequency
          double * y,
    cudaStream_t stream = 0    // cuda stream
);
#endif  // JAMS_CUDA_ARRAY_KERNELS_H
