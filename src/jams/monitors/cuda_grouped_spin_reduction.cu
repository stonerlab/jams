// cuda_grouped_spin_reduction.cu                                      -*-C++-*-
#include <jams/monitors/cuda_grouped_spin_reduction.h>

#if HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include <jams/core/lattice.h>
#include <jams/cuda/cuda_common.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/helpers/error.h>

namespace {

constexpr int kChunkSize =
    static_cast<int>(jams::monitors::kCudaGroupedSpinReductionChunkSize);

int checked_int_count_runtime(const std::size_t count, const char* quantity) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(quantity) + " exceeds int range");
  }
  return static_cast<int>(count);
}

void copy_int_vector_to_device_only(
    jams::MultiArray<int, 1>& target,
    const std::vector<int>& values) {
  target.resize(values.size());
  if (values.empty()) {
    return;
  }

  auto target_values = target.mutable_host_span();
  std::copy(values.begin(), values.end(), target_values.begin());
  target.device_data();
  target.release_stale_host();
}

jams::monitors::CudaSpinGroupChunks make_chunks_from_group_indices(
    const std::vector<std::vector<int>>& group_indices,
    const char* quantity_name) {
  jams::monitors::CudaSpinGroupChunks chunks;
  chunks.num_groups = checked_int_count_runtime(
      group_indices.size(),
      "number of cuda spin reduction groups");

  std::vector<int> all_spin_indices;
  std::vector<int> all_chunk_begin_offsets;
  std::vector<int> all_chunk_end_offsets;
  std::vector<int> all_chunk_group_indices;
  std::vector<int> group_chunk_begin_offsets;
  std::vector<int> group_chunk_end_offsets;

  std::size_t total_spins = 0;
  for (const auto& group : group_indices) {
    total_spins += group.size();
  }

  all_spin_indices.reserve(total_spins);
  group_chunk_begin_offsets.reserve(group_indices.size());
  group_chunk_end_offsets.reserve(group_indices.size());
  const std::string chunk_begin_quantity = std::string(quantity_name) + " chunk begin";
  const std::string chunk_end_quantity = std::string(quantity_name) + " chunk end";
  const std::string chunk_group_quantity = std::string(quantity_name) + " chunk group index";

  for (std::size_t group_index = 0; group_index < group_indices.size(); ++group_index) {
    group_chunk_begin_offsets.push_back(checked_int_count_runtime(
        all_chunk_group_indices.size(),
        "cuda spin reduction group chunk begin"));

    const auto group_begin = all_spin_indices.size();
    all_spin_indices.insert(
        all_spin_indices.end(),
        group_indices[group_index].begin(),
        group_indices[group_index].end());
    const auto group_end = all_spin_indices.size();

    for (std::size_t chunk_begin = group_begin;
         chunk_begin < group_end;
         chunk_begin += jams::monitors::kCudaGroupedSpinReductionChunkSize) {
      const auto chunk_end = std::min(
          chunk_begin + jams::monitors::kCudaGroupedSpinReductionChunkSize,
          group_end);
      all_chunk_begin_offsets.push_back(checked_int_count_runtime(
          chunk_begin,
          chunk_begin_quantity.c_str()));
      all_chunk_end_offsets.push_back(checked_int_count_runtime(
          chunk_end,
          chunk_end_quantity.c_str()));
      all_chunk_group_indices.push_back(checked_int_count_runtime(
          group_index,
          chunk_group_quantity.c_str()));
    }

    group_chunk_end_offsets.push_back(checked_int_count_runtime(
        all_chunk_group_indices.size(),
        "cuda spin reduction group chunk end"));
  }

  chunks.num_chunks = checked_int_count_runtime(
      all_chunk_group_indices.size(),
      "number of cuda spin reduction chunks");

  copy_int_vector_to_device_only(chunks.spin_indices, all_spin_indices);
  copy_int_vector_to_device_only(chunks.chunk_begin_offsets, all_chunk_begin_offsets);
  copy_int_vector_to_device_only(chunks.chunk_end_offsets, all_chunk_end_offsets);
  copy_int_vector_to_device_only(chunks.chunk_group_indices, all_chunk_group_indices);
  copy_int_vector_to_device_only(chunks.group_chunk_begin_offsets, group_chunk_begin_offsets);
  copy_int_vector_to_device_only(chunks.group_chunk_end_offsets, group_chunk_end_offsets);
  return chunks;
}

__device__ double atomic_add_double(double* address, const double value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
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

__device__ void reduce_xyz_block(double& mx, double& my, double& mz) {
  const int thread = threadIdx.x;
  __shared__ double shared_mx[kChunkSize];
  __shared__ double shared_my[kChunkSize];
  __shared__ double shared_mz[kChunkSize];
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

  mx = shared_mx[0];
  my = shared_my[0];
  mz = shared_mz[0];
}

__global__ void grouped_spin_moment_reduction_kernel(
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

  reduce_xyz_block(mx, my, mz);

  if (thread == 0) {
    const int group = chunk_group_indices[chunk];
    atomic_add_double(&group_magnetisation[3 * group + 0], mx);
    atomic_add_double(&group_magnetisation[3 * group + 1], my);
    atomic_add_double(&group_magnetisation[3 * group + 2], mz);
  }
}

__global__ void grouped_spin_sum_chunk_kernel(
    const int num_chunks,
    const int* __restrict__ chunk_begin_offsets,
    const int* __restrict__ chunk_end_offsets,
    const int* __restrict__ spin_indices,
    const double* __restrict__ spins,
    double* __restrict__ chunk_sums) {
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
    mx += spins[3 * spin_index + 0];
    my += spins[3 * spin_index + 1];
    mz += spins[3 * spin_index + 2];
  }

  reduce_xyz_block(mx, my, mz);

  if (thread == 0) {
    chunk_sums[3 * chunk + 0] = mx;
    chunk_sums[3 * chunk + 1] = my;
    chunk_sums[3 * chunk + 2] = mz;
  }
}

__global__ void grouped_spin_sum_finalize_kernel(
    const int num_groups,
    const int* __restrict__ group_chunk_begin_offsets,
    const int* __restrict__ group_chunk_end_offsets,
    const double* __restrict__ chunk_sums,
    double* __restrict__ group_sum) {
  const int group = blockIdx.x;
  if (group >= num_groups) {
    return;
  }

  const int thread = threadIdx.x;
  const int begin = group_chunk_begin_offsets[group];
  const int end = group_chunk_end_offsets[group];

  double mx = 0.0;
  double my = 0.0;
  double mz = 0.0;
  for (int chunk = begin + thread; chunk < end; chunk += blockDim.x) {
    mx += chunk_sums[3 * chunk + 0];
    my += chunk_sums[3 * chunk + 1];
    mz += chunk_sums[3 * chunk + 2];
  }

  reduce_xyz_block(mx, my, mz);

  if (thread == 0) {
    group_sum[3 * group + 0] = mx;
    group_sum[3 * group + 1] = my;
    group_sum[3 * group + 2] = mz;
  }
}

}  // namespace

namespace jams::monitors {

CudaSpinGroupChunks make_cuda_spin_group_chunks(const std::vector<SpinGroup>& groups) {
  std::vector<std::vector<int>> group_indices(groups.size());
  for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
    const auto indices = groups[group_index].indices_span();
    group_indices[group_index].assign(indices.begin(), indices.end());
  }
  return make_chunks_from_group_indices(group_indices, "cuda spin group reduction");
}

CudaSpinGroupChunks make_cuda_basis_spin_group_chunks(
    const Lattice& lattice,
    const int num_spins) {
  std::vector<std::vector<int>> group_indices(
      static_cast<std::size_t>(lattice.num_basis_sites()));
  for (int spin_index = 0; spin_index < num_spins; ++spin_index) {
    const int basis = lattice.lattice_site_basis_index(spin_index);
    if (basis < 0 || basis >= lattice.num_basis_sites()) {
      throw std::runtime_error("invalid basis index while building CUDA spin reduction groups");
    }
    group_indices[static_cast<std::size_t>(basis)].push_back(spin_index);
  }
  return make_chunks_from_group_indices(group_indices, "cuda basis spin reduction");
}

void execute_cuda_grouped_spin_moment_reduction(
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

  grouped_spin_moment_reduction_kernel<<<num_chunks, kChunkSize, 0, stream.get()>>>(
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

void execute_cuda_grouped_spin_sum_reduction(
    CudaStream& stream,
    const int num_groups,
    const int num_chunks,
    const int* chunk_begin_offsets,
    const int* chunk_end_offsets,
    const int* group_chunk_begin_offsets,
    const int* group_chunk_end_offsets,
    const int* spin_indices,
    const double* spins,
    double* chunk_sums,
    double* group_sum) {
  if (num_chunks > 0) {
    grouped_spin_sum_chunk_kernel<<<num_chunks, kChunkSize, 0, stream.get()>>>(
        num_chunks,
        chunk_begin_offsets,
        chunk_end_offsets,
        spin_indices,
        spins,
        chunk_sums);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
  }

  if (num_groups == 0) {
    return;
  }

  grouped_spin_sum_finalize_kernel<<<num_groups, kChunkSize, 0, stream.get()>>>(
      num_groups,
      group_chunk_begin_offsets,
      group_chunk_end_offsets,
      chunk_sums,
      group_sum);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

}  // namespace jams::monitors

#endif

// ----------------------------- END-OF-FILE ----------------------------------
