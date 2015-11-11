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

#endif  // JAMS_CUDA_ARRAY_KERNELS_H
