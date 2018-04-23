#include "cuda_defs.h"

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

__global__ void cuda_array_double_to_float_kernel(
    const unsigned int n,            // n elements in i index
    const double * in,
    float * out
)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = float(in[idx]);
    }
}

__global__ void cuda_array_float_to_double_kernel(
    const unsigned int n,            // n elements in i index
    const float * in,
    double * out
) 
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = double(in[idx]);
    }
}

void cuda_array_double_to_float(
    const unsigned int n,            // n elements in i index
    const double * in,
    float * out,  
    cudaStream_t stream     // cuda stream
)
{
    dim3 block_size;
    block_size.x = 128;

    dim3 grid_size;
    grid_size.x = (n + block_size.x - 1) / block_size.x;

    cuda_array_double_to_float_kernel<<<grid_size, block_size, 0, stream>>>(n, in, out);
}

void cuda_array_float_to_double(
    const unsigned int n,            // n elements in i index
    const float * in,
    double * out,  
    cudaStream_t stream     // cuda stream
) 
{
    dim3 block_size;
    block_size.x = 128;

    dim3 grid_size;
    grid_size.x = (n + block_size.x - 1) / block_size.x;

    cuda_array_float_to_double_kernel<<<grid_size, block_size, 0, stream>>>(n, in, out);
}
