#include "core/cuda_defs.h"

#include <iostream>

// y_ij <-- alpha_i * beta * x_ij

__global__ void cuda_array_elementwise_scale_kernel_general_(
    const unsigned int n,            // n elements in i index
    const unsigned int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const unsigned int incx,         // input increment
    double * y,             // output array
    const unsigned int incy)         // output increment
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n) {
        if (idy < m) {
            y[idx * m + idy + incy] = alpha[idx] * beta * x[idx * m + idy + incx];
        }
    }
}

__global__ void cuda_array_elementwise_scale_kernel_noinc_(
    const unsigned int n,            // n elements in i index
    const unsigned int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    double * y)             // output array
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n) {
        if (idy < m) {
            y[idx * m + idy] = alpha[idx] * beta * x[idx * m + idy];
        }
    }
}

__global__ void cuda_array_elementwise_scale_kernel_noinc_self_(
    const unsigned int n,            // n elements in i index
    const unsigned int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x)             // input/output array
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n) {
        if (idy < m) {
            x[idx * m + idy] = alpha[idx] * beta * x[idx * m + idy];
        }
    }
}

__global__ void cuda_array_elementwise_daxpy_kernel_general_(
    const unsigned int n,            // n elements in i index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const unsigned int incx,         // input increment
    double * y,             // output array
    const unsigned int incy)         // output increment
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx + incy] += alpha[idx] * beta * x[idx + incx];
    }
}

__global__ void cuda_array_elementwise_daxpy_kernel_noinc_(
    const unsigned int n,            // n elements in i index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    double * y)             // output array
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
            y[idx] += alpha[idx] * beta * x[idx];
    }
}

__global__ void cuda_array_remapping_kernel_(
    const unsigned int n,   // n elements in array
    const int * map,          // remapping array
    const double * x,
          double * y)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = x[map[idx]];
    }
}

__global__ void cuda_array_initialize_kernel_(
    const size_t n,   // n elements in array
    double * x,
    const double value)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        x[idx] = value;
    }
}


__global__ void cuda_array_elementwise_cos_kernel_(
    const size_t n,            // n elements in i index
    const double * theta,  // array frequency
    const double   phi,     // constant frequency
          double * y)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = cos(theta[idx] + phi);
    }
}




void cuda_array_elementwise_scale(
    const unsigned int n,            // n elements in i index
    const unsigned int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const unsigned int incx,         // input increment
    double * y,             // output array
    const unsigned int incy,         // output increment
    cudaStream_t stream = 0    // cuda stream
)
{
    dim3 block_size;
    block_size.x = 32;

    // if (m < 4) {
    //     block_size.y = m;
    // } else {
        block_size.y = 4;
    // }

    dim3 grid_size;
    grid_size.x = (n + block_size.x - 1) / block_size.x;
    grid_size.y = (m + block_size.y - 1) / block_size.y;

    if (incx == 1 && incy == 1) {
        if (x == y) {
            cuda_array_elementwise_scale_kernel_noinc_self_<<<grid_size, block_size, 0, stream>>>(n, m, alpha, beta, x);
            cuda_kernel_error_check();
            return;
        } else {
            cuda_array_elementwise_scale_kernel_noinc_<<<grid_size, block_size, 0, stream>>>(n, m, alpha, beta, x, y);
            cuda_kernel_error_check();
            return;
        }
    }

    cuda_array_elementwise_scale_kernel_general_<<<grid_size, block_size, 0, stream>>>(n, m, alpha, beta, x, incx, y, incy);
    cuda_kernel_error_check();
    return;
}

void cuda_array_elementwise_daxpy(
    const unsigned int n,            // n elements in i index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const unsigned int incx,         // input increment
    double * y,             // output array
    const unsigned int incy,         // output increment
    cudaStream_t stream = 0    // cuda stream
)
{
    unsigned int block_size = 1024;

    if (incx == 1 && incy == 1) {
        cuda_array_elementwise_daxpy_kernel_noinc_<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(n, alpha, beta, x, y);
        cuda_kernel_error_check();
        return;
    }

    cuda_array_elementwise_daxpy_kernel_general_<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(n, alpha, beta, x, incx, y, incy);
    cuda_kernel_error_check();
    return;
}


void cuda_array_elementwise_cos(
    const size_t n,            // n elements in i index
    const double * theta,  // array frequency
    const double   phi,    // constant frequency
          double * y,
    cudaStream_t stream = 0    // cuda stream
)
{
    size_t block_size = 1024;

    cuda_array_elementwise_cos_kernel_<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(n, theta, phi, y);
    cuda_kernel_error_check();
}

void cuda_array_remapping(
    const unsigned int n,   // n elements in array
    const int * map,          // remapping array
    const double * x,
          double * y,
    cudaStream_t stream     // cuda stream
) {
    unsigned int block_size = 1024;
    cuda_array_remapping_kernel_<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(n, map, x, y);
}

void cuda_array_initialize(
    const size_t n,   // n elements in array
    double * x,
    const double value,
    cudaStream_t stream     // cuda stream
) {
    size_t block_size = 1024;
    cuda_array_initialize_kernel_<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(n, x, value);
}