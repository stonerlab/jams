// cuda_magnetisation_kernel.cu                                       -*-C++-*-
#include <jams/monitors/cuda_magnetisation_kernel.h>

#include <cuda_runtime.h>

#include <cstddef>

#include <jams/cuda/cuda_common.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/helpers/error.h>

namespace {
constexpr int kMagnetisationChunkSize = 256;

__device__ double atomic_add_double(double* address, const double value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
  // Native double atomicAdd is only available from sm_60.
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  auto old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
#else
  return atomicAdd(address, value);
#endif
}

__global__ void magnetisation_kernel(
    const int num_chunks,
    const int* __restrict__ chunk_begin_offsets,
    const int* __restrict__ chunk_end_offsets,
    const int* __restrict__ chunk_group_indices,
    const int* __restrict__ spin_indices,
    const double* __restrict__ spins,
    const jams::Real* __restrict__ moments,
    double* __restrict__ group_magnetisation) {
  const int chunk = blockIdx.x;
  if (chunk >= num_chunks) {
    return;
  }

  const int thread = threadIdx.x;
  const int begin = chunk_begin_offsets[chunk];
  const int end = chunk_end_offsets[chunk];

  double mx = 0.0;
  double my = 0.0;
  double mz = 0.0;

  for (int offset = begin + thread; offset < end; offset += blockDim.x) {
    const int spin_index = spin_indices[offset];
    const double moment = static_cast<double>(moments[spin_index]);
    mx += moment * spins[3 * spin_index + 0];
    my += moment * spins[3 * spin_index + 1];
    mz += moment * spins[3 * spin_index + 2];
  }

  __shared__ double shared_mx[kMagnetisationChunkSize];
  __shared__ double shared_my[kMagnetisationChunkSize];
  __shared__ double shared_mz[kMagnetisationChunkSize];
  shared_mx[thread] = mx;
  shared_my[thread] = my;
  shared_mz[thread] = mz;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      shared_mx[thread] += shared_mx[thread + stride];
      shared_my[thread] += shared_my[thread + stride];
      shared_mz[thread] += shared_mz[thread + stride];
    }
    __syncthreads();
  }

  if (thread == 0) {
    const int group = chunk_group_indices[chunk];
    atomic_add_double(&group_magnetisation[3 * group + 0], shared_mx[0]);
    atomic_add_double(&group_magnetisation[3 * group + 1], shared_my[0]);
    atomic_add_double(&group_magnetisation[3 * group + 2], shared_mz[0]);
  }
}
}  // namespace

void execute_cuda_magnetisation_kernel(
    CudaStream& stream,
    const int num_groups,
    const int num_chunks,
    const int* chunk_begin_offsets,
    const int* chunk_end_offsets,
    const int* chunk_group_indices,
    const int* spin_indices,
    const double* spins,
    const jams::Real* moments,
    double* group_magnetisation) {
  if (num_groups > 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(
        group_magnetisation,
        0,
        static_cast<std::size_t>(num_groups) * 3 * sizeof(double),
        stream.get()));
  }

  if (num_chunks == 0) {
    return;
  }

  dim3 block_size(kMagnetisationChunkSize, 1, 1);
  dim3 grid_size(num_chunks, 1, 1);

  magnetisation_kernel<<<grid_size, block_size, 0, stream.get()>>>(
      num_chunks,
      chunk_begin_offsets,
      chunk_end_offsets,
      chunk_group_indices,
      spin_indices,
      spins,
      moments,
      group_magnetisation);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

// ----------------------------- END-OF-FILE ----------------------------------
