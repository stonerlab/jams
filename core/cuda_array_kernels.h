// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_ARRAY_KERNELS_H
#define JAMS_CUDA_ARRAY_KERNELS_H

void cuda_array_elementwise_scale(
    const int n,            // n elements in i index
    const int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const int incx,         // input increment
    double * y,             // output array
    const int incy          // output increment
);

#endif  // JAMS_CUDA_ARRAY_KERNELS_H
