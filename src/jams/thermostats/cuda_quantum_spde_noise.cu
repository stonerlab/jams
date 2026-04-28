// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/cuda_quantum_spde_noise.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <jams/common.h>

#include "jams/cuda/cuda_common.h"
#include "jams/helpers/consts.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde_kernel.cuh"

namespace {

void generate_normal(jams::MultiArray<jams::Real, 1>& data) {
#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(
      jams::instance().curand_generator(), data.mutable_device_data(), data.size(), 0.0, 1.0));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(
      jams::instance().curand_generator(), data.mutable_device_data(), data.size(), 0.0, 1.0));
#endif
}

void reset_curand_stream_to_default() {
  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), nullptr));
}

void solve_3x3(double a[3][4]) {
  for (auto pivot = 0; pivot < 3; ++pivot) {
    auto pivot_row = pivot;
    auto pivot_abs = fabs(a[pivot][pivot]);
    for (auto row = pivot + 1; row < 3; ++row) {
      const auto row_abs = fabs(a[row][pivot]);
      if (row_abs > pivot_abs) {
        pivot_abs = row_abs;
        pivot_row = row;
      }
    }

    if (pivot_abs == 0.0) {
      throw std::runtime_error("singular stationary covariance system in quantum SPDE thermostat");
    }

    if (pivot_row != pivot) {
      for (auto col = pivot; col < 4; ++col) {
        std::swap(a[pivot][col], a[pivot_row][col]);
      }
    }

    const double inv_pivot = 1.0 / a[pivot][pivot];
    for (auto col = pivot; col < 4; ++col) {
      a[pivot][col] *= inv_pivot;
    }

    for (auto row = 0; row < 3; ++row) {
      if (row == pivot) {
        continue;
      }

      const double factor = a[row][pivot];
      for (auto col = pivot; col < 4; ++col) {
        a[row][col] -= factor * a[pivot][col];
      }
    }
  }
}

}  // namespace

namespace jams {

void quantum_spde_bose_exact_update_host(const double gamma, const double omega,
                                         const double eta0, const double h,
                                         double z[2]) {
  const double omega2 = omega * omega;
  const double alpha = 0.5 * gamma;
  const double decay = exp(-alpha * h);
  const double force_eq = eta0 / omega2;

  double y0 = z[0] - force_eq;
  double v0 = z[1];

  const double discriminant = omega2 - alpha * alpha;
  if (discriminant > 0.0) {
    const double beta = sqrt(discriminant);
    const double c = cos(beta * h);
    const double s = sin(beta * h);
    const double inv_beta = 1.0 / beta;

    const double y1 = decay * (y0 * c + (v0 + alpha * y0) * inv_beta * s);
    const double v1 = decay * (v0 * c - (alpha * v0 + omega2 * y0) * inv_beta * s);

    z[0] = y1 + force_eq;
    z[1] = v1;
    return;
  }

  if (discriminant < 0.0) {
    const double beta = sqrt(-discriminant);
    const double c = cosh(beta * h);
    const double s = sinh(beta * h);
    const double inv_beta = 1.0 / beta;

    const double y1 = decay * (y0 * c + (v0 + alpha * y0) * inv_beta * s);
    const double v1 = decay * (v0 * c - (alpha * v0 + omega2 * y0) * inv_beta * s);

    z[0] = y1 + force_eq;
    z[1] = v1;
    return;
  }

  const double y1 = decay * (y0 + (v0 + alpha * y0) * h);
  const double v1 = decay * (v0 - alpha * (v0 + alpha * y0) * h);

  z[0] = y1 + force_eq;
  z[1] = v1;
}

QuantumSpdeBoseCholesky quantum_spde_stationary_bose_cholesky(
    const double gamma, const double omega, const double h) {
  if (h <= 0.0) {
    return {};
  }

  double z[2] = {1.0, 0.0};
  quantum_spde_bose_exact_update_host(gamma, omega, 0.0, h, z);
  const double a00 = z[0];
  const double a10 = z[1];

  z[0] = 0.0;
  z[1] = 1.0;
  quantum_spde_bose_exact_update_host(gamma, omega, 0.0, h, z);
  const double a01 = z[0];
  const double a11 = z[1];

  z[0] = 0.0;
  z[1] = 0.0;
  quantum_spde_bose_exact_update_host(gamma, omega, 1.0, h, z);
  const double b0 = z[0];
  const double b1 = z[1];

  const double force_variance = 2.0 * gamma / h;
  const double q00 = force_variance * b0 * b0;
  const double q01 = force_variance * b0 * b1;
  const double q11 = force_variance * b1 * b1;

  double system[3][4] = {
      {1.0 - a00 * a00, -2.0 * a00 * a01, -a01 * a01, q00},
      {-a00 * a10, 1.0 - (a00 * a11 + a01 * a10), -a01 * a11, q01},
      {-a10 * a10, -2.0 * a10 * a11, 1.0 - a11 * a11, q11},
  };

  solve_3x3(system);

  const double p00 = system[0][3];
  const double p01 = system[1][3];
  const double p11 = system[2][3];
  if (p00 <= 0.0) {
    throw std::runtime_error("non-positive stationary position variance in quantum SPDE thermostat");
  }

  QuantumSpdeBoseCholesky factor;
  factor.l00 = sqrt(p00);
  factor.l10 = p01 / factor.l00;
  factor.l11 = sqrt(std::max(0.0, p11 - factor.l10 * factor.l10));
  return factor;
}

CudaQuantumSpdeNoiseGenerator::CudaQuantumSpdeNoiseGenerator(
    const int process_count, const double delta_tau, const double omega_max,
    const bool zero_point, CudaStream& update_stream)
    : process_count_(process_count),
      delta_tau_(delta_tau),
      omega_max_(omega_max),
      zero_point_(zero_point),
      update_stream_(update_stream) {
  cudaEventCreateWithFlags(&curand_done_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  cudaEventCreateWithFlags(&eta1a_reusable_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  cudaEventCreateWithFlags(&eta1b_reusable_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  cudaEventCreateWithFlags(&eta0a_reusable_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  cudaEventCreateWithFlags(&eta0b_reusable_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cudaEventRecord(eta1a_reusable_, update_stream_.get());
  cudaEventRecord(eta1b_reusable_, update_stream_.get());
  cudaEventRecord(eta0a_reusable_, update_stream_.get());
  cudaEventRecord(eta0b_reusable_, update_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  zeta5_.resize(process_count_).zero();
  zeta5p_.resize(process_count_).zero();
  zeta6_.resize(process_count_).zero();
  zeta6p_.resize(process_count_).zero();
  eta1a_.resize(2 * process_count_).zero();
  eta1b_.resize(2 * process_count_).zero();

  if (zero_point_) {
    zeta0_.resize(4 * process_count_).zero();
    eta0a_.resize(4 * process_count_).zero();
    eta0b_.resize(4 * process_count_).zero();
  }

  generate_random_buffers();
}

CudaQuantumSpdeNoiseGenerator::~CudaQuantumSpdeNoiseGenerator() {
  synchronize();

  if (eta0b_reusable_ != nullptr) {
    cudaEventDestroy(eta0b_reusable_);
    eta0b_reusable_ = nullptr;
  }

  if (eta0a_reusable_ != nullptr) {
    cudaEventDestroy(eta0a_reusable_);
    eta0a_reusable_ = nullptr;
  }

  if (eta1b_reusable_ != nullptr) {
    cudaEventDestroy(eta1b_reusable_);
    eta1b_reusable_ = nullptr;
  }

  if (eta1a_reusable_ != nullptr) {
    cudaEventDestroy(eta1a_reusable_);
    eta1a_reusable_ = nullptr;
  }

  if (curand_done_ != nullptr) {
    cudaEventDestroy(curand_done_);
    curand_done_ = nullptr;
  }
}

void CudaQuantumSpdeNoiseGenerator::generate_random_buffers() {
  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
  if (zero_point_) {
    generate_normal(eta0a_);
    generate_normal(eta0b_);
  }
  generate_normal(eta1a_);
  generate_normal(eta1b_);

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  reset_curand_stream_to_default();
}

void CudaQuantumSpdeNoiseGenerator::zero_state() {
  synchronize();

  zeta5_.zero();
  zeta5p_.zero();
  zeta6_.zero();
  zeta6p_.zero();
  if (zero_point_) {
    zeta0_.zero();
  }
}

void CudaQuantumSpdeNoiseGenerator::initialize(const Initialization initialization,
                                              const jams::Real temperature) {
  if (initialization == Initialization::Stationary) {
    initialize_stationary(temperature);
    return;
  }

  zero_state();
  generate_random_buffers();
}

void CudaQuantumSpdeNoiseGenerator::initialize_stationary(const jams::Real temperature) {
  zero_state();

  int block_size = 128;
  int grid_size = (process_count_ + block_size - 1) / block_size;
  bool consumed_random_numbers = false;

  cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);

  const double reduced_delta_tau = delta_tau_ * temperature;
  if (reduced_delta_tau > 0.0) {
    const auto factor5 = quantum_spde_stationary_bose_cholesky(5.0142, 2.7189, reduced_delta_tau);
    const auto factor6 = quantum_spde_stationary_bose_cholesky(3.2974, 1.2223, reduced_delta_tau);
    cuda_thermostat_quantum_spde_stationary_no_zero_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
        zeta6p_.mutable_device_data(), eta1a_.device_data(), eta1b_.device_data(),
        factor5.l00, factor5.l10, factor5.l11, factor6.l00, factor6.l10, factor6.l11,
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    consumed_random_numbers = true;
  }

  if (zero_point_) {
    const double zero_point_delta_tau = (kHBarIU * omega_max_ * delta_tau_) / kBoltzmannIU;
    cuda_thermostat_quantum_spde_stationary_zero_point_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        zeta0_.mutable_device_data(), eta0a_.device_data(), static_cast<jams::Real>(zero_point_delta_tau),
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    consumed_random_numbers = true;
  }

  if (consumed_random_numbers) {
    update_stream_.synchronize();
    generate_random_buffers();
  }
}

void CudaQuantumSpdeNoiseGenerator::warmup(const unsigned steps,
                                          const jams::Real temperature,
                                          jams::Real* noise,
                                          const jams::Real* sigma) {
  for (auto i = 0u; i < steps; ++i) {
    update(noise, sigma, temperature);
  }
}

void CudaQuantumSpdeNoiseGenerator::update(jams::Real* noise,
                                           const jams::Real* sigma,
                                           const jams::Real temperature) {
  int block_size = 128;
  int grid_size = (process_count_ + block_size - 1) / block_size;

  const double zero_point_delta_tau = (kHBarIU * omega_max_ * delta_tau_) / kBoltzmannIU;
  const double zero_point_scale = (kHBarIU * omega_max_) / kBoltzmannIU;

  if (temperature == 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise, 0, process_count_ * sizeof(jams::Real), update_stream_.get()));

    if (zero_point_) {
      swap(eta0a_, eta0b_);
      std::swap(eta0a_reusable_, eta0b_reusable_);

      cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
      cudaStreamWaitEvent(curand_stream_.get(), eta0a_reusable_, 0);
      cuda_thermostat_quantum_spde_zero_point_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
          noise, zeta0_.mutable_device_data(), eta0b_.device_data(), sigma,
          static_cast<jams::Real>(zero_point_delta_tau), static_cast<jams::Real>(zero_point_scale),
          process_count_);
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
      cudaEventRecord(eta0b_reusable_, update_stream_.get());
      DEBUG_CHECK_CUDA_ASYNC_STATUS

      CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
      generate_normal(eta0a_);

      cudaEventRecord(curand_done_, curand_stream_.get());
      DEBUG_CHECK_CUDA_ASYNC_STATUS
      reset_curand_stream_to_default();
    }

    return;
  }

  const double reduced_delta_tau = delta_tau_ * temperature;

  // The refill target was used by an earlier update before the swap. Each
  // buffer carries its own reusable event, so CURAND only waits for the buffer
  // it is about to overwrite.
  swap(eta1a_, eta1b_);
  std::swap(eta1a_reusable_, eta1b_reusable_);

  cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
  cudaStreamWaitEvent(curand_stream_.get(), eta1a_reusable_, 0);
  cuda_thermostat_quantum_spde_no_zero_kernel<<<grid_size, block_size, 0, update_stream_.get() >>> (
    noise, zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
    zeta6p_.mutable_device_data(), eta1b_.device_data(), sigma, reduced_delta_tau,
    temperature, process_count_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
  cudaEventRecord(eta1b_reusable_, update_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
  generate_normal(eta1a_);

  if (zero_point_) {
    swap(eta0a_, eta0b_);
    std::swap(eta0a_reusable_, eta0b_reusable_);
    cudaStreamWaitEvent(curand_stream_.get(), eta0a_reusable_, 0);

    cuda_thermostat_quantum_spde_zero_point_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        noise, zeta0_.mutable_device_data(), eta0b_.device_data(), sigma,
        static_cast<jams::Real>(zero_point_delta_tau), static_cast<jams::Real>(zero_point_scale),
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    cudaEventRecord(eta0b_reusable_, update_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS

    CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
    generate_normal(eta0a_);
  }

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  reset_curand_stream_to_default();
}

void CudaQuantumSpdeNoiseGenerator::synchronize() {
  update_stream_.synchronize();
  curand_stream_.synchronize();
}

}  // namespace jams
