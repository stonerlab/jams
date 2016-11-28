#include "core/cuda_defs.h"
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>


// y_ij <-- alpha_i * beta * x_ij

template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

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
    const double * x,             // input array
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
    cudaStream_t stream    // cuda stream
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

double cuda_array_sum(
    const unsigned int n,
    const double * x,
    const unsigned int incx,
    cudaStream_t stream) {


    thrust::device_ptr<const double> d = thrust::device_pointer_cast(x);  

    typedef thrust::device_ptr<const double> Iterator;

    strided_range<Iterator> elements(d, d + n, incx);

    return thrust::reduce(thrust::cuda::par.on(stream), elements.begin(), elements.end());
}
