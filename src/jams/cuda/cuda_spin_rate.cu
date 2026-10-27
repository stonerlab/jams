#include "jams/cuda/cuda_solver.h"

#include <stdexcept>
#include "jams/cuda/cuda_common.h"
#include "jams/solvers/llg_spin_rate.h"

namespace {

constexpr int kBlockSize = 256;

__device__ double block_max_rate(double value, double* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = jams::solvers::combine_spin_rates(
          shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

__global__ void spin_rate_partials(
    int count, const double* spins, const jams::Real* fields,
    const jams::Real* inverse_moments, const jams::Real* alpha,
    const jams::Real* gyro, double* partials) {
  __shared__ double shared[kBlockSize];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double value = 0.0;
  if (i < count) {
    value = jams::solvers::llg_spin_rate(
        spins + 3 * i, fields ? fields + 3 * i : nullptr,
        inverse_moments[i], alpha[i], gyro[i]);
  }
  const double maximum = block_max_rate(value, shared);
  if (threadIdx.x == 0) partials[blockIdx.x] = maximum;
}

__global__ void finish_spin_rate(int count, const double* partials, double* result) {
  __shared__ double shared[kBlockSize];
  double value = 0.0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    value = jams::solvers::combine_spin_rates(value, partials[i]);
  }
  const double maximum = block_max_rate(value, shared);
  if (threadIdx.x == 0) result[0] = maximum;
}

}  // namespace

double CudaSolver::max_llg_spin_rate_cuda(const jams::MultiArray<jams::Real, 1>& gyro_eff) {
  if (globals::num_magnetic_spins == 0) {
    throw std::runtime_error("spin-rate requires at least one magnetic spin");
  }
  auto stream = jams::instance().cuda_master_stream().get();
  wait_on_spin_barrier_event(stream);
#if DO_MIXED_PRECISION
  // Physics may have changed spins since the solver populated this cache.
  // Rebuild it on the master stream before evaluating the diagnostic fields.
  field_spin_cache_valid_ = false;
#endif
  compute_fields();  // Orders completed Hamiltonian fields onto the master stream.
  const int blocks = 1 + (globals::num_spins - 1) / kBlockSize;
  if (spin_rate_partials_.size() != blocks) spin_rate_partials_.resize(blocks);
  if (spin_rate_result_.size() != 1) spin_rate_result_.resize(1);

  spin_rate_partials<<<blocks, kBlockSize, 0, stream>>>(
      globals::num_spins, globals::s.device_data(),
      hamiltonians_.empty() ? nullptr : globals::h.device_data(),
      globals::inv_mus.device_data(), globals::alpha.device_data(),
      gyro_eff.device_data(), spin_rate_partials_.mutable_device_data());
  CHECK_CUDA_STATUS(cudaGetLastError());
  finish_spin_rate<<<1, kBlockSize, 0, stream>>>(
      blocks, spin_rate_partials_.device_data(), spin_rate_result_.mutable_device_data());
  CHECK_CUDA_STATUS(cudaGetLastError());

  double result;
  CHECK_CUDA_STATUS(cudaMemcpyAsync(&result, spin_rate_result_.device_data(),
                                  sizeof(result), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_STATUS(cudaStreamSynchronize(stream));
  if (result < 0.0 || !jams::is_finite(result)) {
    throw std::runtime_error("non-finite input or result in deterministic spin rate");
  }
  return result;
}
