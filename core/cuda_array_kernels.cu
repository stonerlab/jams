#include "core/cuda_defs.h"

// y_ij <-- alpha_i * beta * x_ij

__global__ void cuda_array_elementwise_scale_kernel_general_(
    const int n,            // n elements in i index
    const int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const int incx,         // input increment
    double * y,             // output array
    const int incy)         // output increment
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n) {
        if (idy < m) {
            y[idx * m + idy + incy] = alpha[idx] * beta * x[idx * m + idy + incx];
        }
    }
}

__global__ void cuda_array_elementwise_scale_kernel_noinc_(
    const int n,            // n elements in i index
    const int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    double * y)             // output array
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n) {
        if (idy < m) {
            y[idx * m + idy] = alpha[idx] * beta * x[idx * m + idy];
        }
    }
}

__global__ void cuda_array_elementwise_scale_kernel_noinc_self_(
    const int n,            // n elements in i index
    const int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x)             // input/output array
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n) {
        if (idy < m) {
            x[idx * m + idy] = alpha[idx] * beta * x[idx * m + idy];
        }
    }
}

void cuda_array_elementwise_scale(
    const int n,            // n elements in i index
    const int m,            // m elements in j index
    const double * alpha,   // scale factors array of length n
    const double   beta,    // uniform scale factor
    double * x,             // input array
    const int incx,         // input increment
    double * y,             // output array
    const int incy          // output increment
)
{
    dim3 block_size;
    block_size.x = 16;

    if (m < 4) {
        block_size.y = m;
    } else {
        block_size.y = 4;
    }

    dim3 grid_size;
    grid_size.x = (n + block_size.x - 1) / block_size.x;
    grid_size.y = (m + block_size.y - 1) / block_size.y;

    if (incx == 1 && incy == 1) {
        if (x == y) {
            cuda_array_elementwise_scale_kernel_noinc_self_<<<grid_size, block_size>>>(n, m, alpha, beta, x);
            cuda_kernel_error_check();
            return;
        } else {
            cuda_array_elementwise_scale_kernel_noinc_<<<grid_size, block_size>>>(n, m, alpha, beta, x, y);
            cuda_kernel_error_check();
            return;
        }
    }

    cuda_array_elementwise_scale_kernel_general_<<<grid_size, block_size>>>(n, m, alpha, beta, x, incx, y, incy);
    cuda_kernel_error_check();
    return;
}


