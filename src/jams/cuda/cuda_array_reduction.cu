#include <jams/cuda/cuda_array_reduction.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>

#include "jams/containers/multiarray.h"
#include "jams/helpers/array_ops.h"

#define MAX_THREADS 256

unsigned int next_pow2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

template <unsigned int blockSize, typename T>
__device__ void
in_thread_reduction(int tid, T *sdata, T& mySum) {
  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 256].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 256].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 256].z;
  }

  __syncthreads();

  if ((blockSize >= 256) &&(tid < 128))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 128].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 128].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 128].z;
  }

  __syncthreads();

  if ((blockSize >= 128) && (tid <  64))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 64].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 64].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 64].z;
  }

  __syncthreads();

  if ( tid < 32 ) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >=  64) {
      mySum.x += sdata[tid + 32].x;
      mySum.y += sdata[tid + 32].y;
      mySum.z += sdata[tid + 32].z;
    }
    // Reduce final warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
      // Mask 0xFFFFFFFF see: https://stackoverflow.com/questions/50639194/shfl-down-and-shfl-down-sync-give-different-results
      mySum.x += __shfl_down_sync(0xFFFFFFFF, mySum.x, offset);
      mySum.y += __shfl_down_sync(0xFFFFFFFF, mySum.y, offset);
      mySum.z += __shfl_down_sync(0xFFFFFFFF, mySum.z, offset);
    }
  }
}

template <unsigned int blockSize, typename T>
__global__ void
vector_field_reduce_kernel(const T *g_data, T *g_block_sums, unsigned int n)
{
  using T3 = std::conditional_t<std::is_same_v<T, double>, double3, float3>;
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, "type must be float or double");

  extern __shared__ unsigned char sdata_raw[];
  T3* sdata = reinterpret_cast<T3*>(sdata_raw);

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

  T3 mySum = {0, 0, 0};
  if (i < n) {
    mySum.x = static_cast<T>(g_data[3 * i + 0]);
    mySum.y = static_cast<T>(g_data[3 * i + 1]);
    mySum.z = static_cast<T>(g_data[3 * i + 2]);
  }

  if (i + blockSize < n) {
    mySum.x += g_data[3*(i + blockSize)+0];
    mySum.y += g_data[3*(i + blockSize)+1];
    mySum.z += g_data[3*(i + blockSize)+2];
  }

  sdata[tid] = mySum;
  __syncthreads();

  in_thread_reduction<blockSize>(tid, sdata, mySum);

  // write result for this block to global mem
  if (tid == 0) {
    g_block_sums[3 * blockIdx.x + 0] = mySum.x;
    g_block_sums[3 * blockIdx.x + 1] = mySum.y;
    g_block_sums[3 * blockIdx.x + 2] = mySum.z;
  }
}


template <unsigned int blockSize, typename T>
__global__ void
vector_field_multiply_and_reduce_kernel(const T *g_data, const T *g_ifactors, T *g_block_sums, unsigned int n)
{
  using T3 = std::conditional_t<std::is_same_v<T, double>, double3, float3>;
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, "type must be float or double");

  extern __shared__ unsigned char sdata_raw[];
  T3* sdata = reinterpret_cast<T3*>(sdata_raw);

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

  T3 mySum = {0, 0, 0};
  if (i < n) {
    mySum.x = g_data[3*i+0] * g_ifactors[i];
    mySum.y = g_data[3*i+1] * g_ifactors[i];
    mySum.z = g_data[3*i+2] * g_ifactors[i];
  }

  if (i + blockSize < n) {
    mySum.x += g_data[3*(i + blockSize)+0] * g_ifactors[(i + blockSize)];
    mySum.y += g_data[3*(i + blockSize)+1] * g_ifactors[(i + blockSize)];
    mySum.z += g_data[3*(i + blockSize)+2] * g_ifactors[(i + blockSize)];
  }

  sdata[tid] = mySum;
  __syncthreads();

  in_thread_reduction<blockSize>(tid, sdata, mySum);

  // write result for this block to global mem
  if (tid == 0) {
    g_block_sums[3 * blockIdx.x + 0] = mySum.x;
    g_block_sums[3 * blockIdx.x + 1] = mySum.y;
    g_block_sums[3 * blockIdx.x + 2] = mySum.z;
  }
}


template <unsigned int blockSize, typename T>
__global__ void
vector_field_indexed_reduce_kernel(const T *g_data, const int *g_indicies, T *g_block_sums, unsigned int n)
{
  using T3 = std::conditional_t<std::is_same_v<T, double>, double3, float3>;
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, "type must be float or double");

  extern __shared__ unsigned char sdata_raw[];
  T3* sdata = reinterpret_cast<T3*>(sdata_raw);

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

  T3 mySum = {0, 0, 0};
  if (i < n) {
    mySum.x = g_data[3*g_indicies[i]+0];
    mySum.y = g_data[3*g_indicies[i]+1];
    mySum.z = g_data[3*g_indicies[i]+2];
  }

  if (i + blockSize < n) {
    mySum.x += g_data[3*g_indicies[i + blockSize]+0];
    mySum.y += g_data[3*g_indicies[i + blockSize]+1];
    mySum.z += g_data[3*g_indicies[i + blockSize]+2];
  }

  sdata[tid] = mySum;
  __syncthreads();

  in_thread_reduction<blockSize>(tid, sdata, mySum);

  // write result for this block to global mem
  if (tid == 0) {
    g_block_sums[3 * blockIdx.x + 0] = mySum.x;
    g_block_sums[3 * blockIdx.x + 1] = mySum.y;
    g_block_sums[3 * blockIdx.x + 2] = mySum.z;
  }
}


template <unsigned int blockSize, typename T1, typename T2>
__global__ void
vector_field_key_multiply_and_reduce_kernel(const T1 *g_data, const T2 *g_ifactors, const int *g_indicies, T1 *g_block_sums, unsigned int n)
{
  using T3 = std::conditional_t<std::is_same_v<T1, double>, double3, float3>;
  static_assert(std::is_same_v<T1, double> || std::is_same_v<T1, float>, "type must be float or double");

  extern __shared__ unsigned char sdata_raw[];
  T3* sdata = reinterpret_cast<T3*>(sdata_raw);

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

  T3 mySum = {0, 0, 0};
  if (i < n) {
    mySum.x = g_data[3*g_indicies[i]+0] * g_ifactors[g_indicies[i]];
    mySum.y = g_data[3*g_indicies[i]+1] * g_ifactors[g_indicies[i]];;
    mySum.z = g_data[3*g_indicies[i]+2] * g_ifactors[g_indicies[i]];;
  }

  if (i + blockSize < n) {
    mySum.x += g_data[3*g_indicies[i + blockSize]+0] * g_ifactors[g_indicies[i + blockSize]];
    mySum.y += g_data[3*g_indicies[i + blockSize]+1] * g_ifactors[g_indicies[i + blockSize]];
    mySum.z += g_data[3*g_indicies[i + blockSize]+2] * g_ifactors[g_indicies[i + blockSize]];
  }

  sdata[tid] = mySum;
  __syncthreads();

  in_thread_reduction<blockSize>(tid, sdata, mySum);

  // write result for this block to global mem
  if (tid == 0) {
    g_block_sums[3 * blockIdx.x + 0] = mySum.x;
    g_block_sums[3 * blockIdx.x + 1] = mySum.y;
    g_block_sums[3 * blockIdx.x + 2] = mySum.z;
  }
}


Vec3 jams::vector_field_reduce_cuda(const jams::MultiArray<double, 2>& x) {
  assert(x.size(1) == 3);

  int size = x.size(0);
  int threads = (size < MAX_THREADS*2) ? next_pow2((size + 1)/ 2) : MAX_THREADS;
  int blocks = (size + (threads * 2 - 1)) / (threads * 2);

  // This is a static buffer so that we don't have to keep reallocating every
  // call. THIS IS NOT THREAD SAFE
  static jams::MultiArray<double, 2> block_sums;

  if (block_sums.size(0) < blocks) {
    block_sums.resize(blocks, 3);
  }

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(double3) : threads * sizeof(double3);

  switch (threads)
  {
    case 512:
      vector_field_reduce_kernel<512><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case 256:
      vector_field_reduce_kernel<256><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case 128:
      vector_field_reduce_kernel<128><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case 64:
      vector_field_reduce_kernel< 64><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case 32:
      vector_field_reduce_kernel< 32><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case 16:
      vector_field_reduce_kernel< 16><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case  8:
      vector_field_reduce_kernel<  8><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case  4:
      vector_field_reduce_kernel<  4><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case  2:
      vector_field_reduce_kernel<  2><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;

    case  1:
      vector_field_reduce_kernel<  1><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), block_sums.device_data(), size);
      break;
  }

  return jams::vector_field_reduce(block_sums);
}


Vec3 jams::vector_field_indexed_reduce_cuda(const jams::MultiArray<double, 2>& x, const jams::MultiArray<int, 1>& indices) {
  assert(x.size(1) == 3);

  int size = indices.size();
  int threads = (size < MAX_THREADS*2) ? next_pow2((size + 1)/ 2) : MAX_THREADS;
  int blocks = (size + (threads * 2 - 1)) / (threads * 2);

  // This is a static buffer so that we don't have to keep reallocating every
  // call. THIS IS NOT THREAD SAFE
  static jams::MultiArray<double, 2> block_sums;

  if (block_sums.size(0) < blocks) {
    block_sums.resize(blocks, 3);
  }

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(double3) : threads * sizeof(double3);

  switch (threads)
  {
    case 512:
      vector_field_indexed_reduce_kernel<512><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 256:
      vector_field_indexed_reduce_kernel<256><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 128:
      vector_field_indexed_reduce_kernel<128><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 64:
      vector_field_indexed_reduce_kernel< 64><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 32:
      vector_field_indexed_reduce_kernel< 32><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 16:
      vector_field_indexed_reduce_kernel< 16><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case  8:
      vector_field_indexed_reduce_kernel<  8><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case  4:
      vector_field_indexed_reduce_kernel<  4><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case  2:
      vector_field_indexed_reduce_kernel<  2><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case  1:
      vector_field_indexed_reduce_kernel<  1><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;
  }

  return jams::vector_field_reduce(block_sums);
}


Vec3 jams::vector_field_scale_and_reduce_cuda(const jams::MultiArray<double, 2>& x, const jams::MultiArray<double, 1>& scale_factors) {
  assert(x.size(1) == 3);
  assert(x.size(0) == scale_factors.size());

  int size = x.size(0);
  int threads = (size < MAX_THREADS*2) ? next_pow2((size + 1)/ 2) : MAX_THREADS;
  int blocks = (size + (threads * 2 - 1)) / (threads * 2);

  // This is a static buffer so that we don't have to keep reallocating every
  // call. THIS IS NOT THREAD SAFE
  static jams::MultiArray<double, 2> block_sums;

  if (block_sums.size(0) < blocks) {
    block_sums.resize(blocks, 3);
  }

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(double3) : threads * sizeof(double3);

  switch (threads)
  {
    case 512:
      vector_field_multiply_and_reduce_kernel<512><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case 256:
      vector_field_multiply_and_reduce_kernel<256><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case 128:
      vector_field_multiply_and_reduce_kernel<128><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case 64:
      vector_field_multiply_and_reduce_kernel< 64><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case 32:
      vector_field_multiply_and_reduce_kernel< 32><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case 16:
      vector_field_multiply_and_reduce_kernel< 16><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case  8:
      vector_field_multiply_and_reduce_kernel<  8><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case  4:
      vector_field_multiply_and_reduce_kernel<  4><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case  2:
      vector_field_multiply_and_reduce_kernel<  2><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;

    case  1:
      vector_field_multiply_and_reduce_kernel<  1><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), block_sums.device_data(), size);
      break;
  }

  return jams::vector_field_reduce(block_sums);
}

template <typename X, typename S>
std::array<X,3> jams::vector_field_indexed_scale_and_reduce_cuda(const jams::MultiArray<X, 2>& x, const jams::MultiArray<S, 1>& scale_factors, const jams::MultiArray<int, 1>& indices) {
  assert(x.size(1) == 3);
  assert(x.size(0) == scale_factors.size());

  using X3 = std::conditional_t<std::is_same_v<X, double>, double3, float3>;
  static_assert(std::is_same_v<X, double> || std::is_same_v<X, float>, "type must be float or double");

  int size = indices.size();
  int threads = (size < MAX_THREADS*2) ? next_pow2((size + 1)/ 2) : MAX_THREADS;
  int blocks = (size + (threads * 2 - 1)) / (threads * 2);

  // This is a static buffer so that we don't have to keep reallocating every
  // call. THIS IS NOT THREAD SAFE
  static jams::MultiArray<X, 2> block_sums;

  if (block_sums.size(0) < blocks) {
    block_sums.resize(blocks, 3);
  }

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(X3) : threads * sizeof(X3);

  switch (threads)
  {
    case 512:
      vector_field_key_multiply_and_reduce_kernel<512><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 256:
      vector_field_key_multiply_and_reduce_kernel<256><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 128:
      vector_field_key_multiply_and_reduce_kernel<128><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 64:
      vector_field_key_multiply_and_reduce_kernel< 64><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 32:
      vector_field_key_multiply_and_reduce_kernel< 32><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case 16:
      vector_field_key_multiply_and_reduce_kernel< 16><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case  8:
      vector_field_key_multiply_and_reduce_kernel<  8><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case  4:
      vector_field_key_multiply_and_reduce_kernel<  4><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case  2:
      vector_field_key_multiply_and_reduce_kernel<  2><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;

    case  1:
      vector_field_key_multiply_and_reduce_kernel<  1><<< dimGrid, dimBlock, smemSize >>>(x.device_data(), scale_factors.device_data(), indices.device_data(), block_sums.device_data(), size);
      break;
  }

  return jams::vector_field_reduce(block_sums);
}

template std::array<double,3> jams::vector_field_indexed_scale_and_reduce_cuda<double,float>(const jams::MultiArray<double, 2>& x, const jams::MultiArray<float, 1>& scale_factors, const jams::MultiArray<int, 1>& indices);

template std::array<float,3> jams::vector_field_indexed_scale_and_reduce_cuda<float,float>(const jams::MultiArray<float, 2>& x, const jams::MultiArray<float, 1>& scale_factors, const jams::MultiArray<int, 1>& indices);

template std::array<double,3> jams::vector_field_indexed_scale_and_reduce_cuda<double,double>(const jams::MultiArray<double, 2>& x, const jams::MultiArray<double, 1>& scale_factors, const jams::MultiArray<int, 1>& indices);


float jams::scalar_field_reduce_cuda(const jams::MultiArray<float, 1> &x, cudaStream_t stream) {
  return thrust::reduce(thrust::cuda::par.on(stream), thrust::device_ptr<const float>(x.device_data()), thrust::device_ptr<const float>(x.device_data() + x.elements()), float(0), thrust::plus<float>());
}

double jams::scalar_field_reduce_cuda(const jams::MultiArray<double, 1> &x, cudaStream_t stream) {
  return thrust::reduce(thrust::cuda::par.on(stream), thrust::device_ptr<const double>(x.device_data()), thrust::device_ptr<const double>(x.device_data() + x.elements()), double(0), thrust::plus<double>());
}














#undef MAX_THREADS