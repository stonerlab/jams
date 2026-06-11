// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/cuda_quantum_spde_noise.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <jams/common.h>

#include "jams/cuda/cuda_common.h"
#include "jams/helpers/consts.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde_kernel.cuh"

namespace {

void generate_normal(curandGenerator_t generator, jams::MultiArray<jams::Real, 1>& data) {
#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(
      generator, data.mutable_device_data(), data.size(), 0.0, 1.0));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(
      generator, data.mutable_device_data(), data.size(), 0.0, 1.0));
#endif
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

QuantumSpdeBoseUpdateCoefficients quantum_spde_bose_update_coefficients(
    const double gamma, const double omega, const double h) {
  if (h <= 0.0) {
    return {};
  }

  const double omega2 = omega * omega;
  const double alpha = 0.5 * gamma;
  const double decay = exp(-alpha * h);
  const double discriminant = omega2 - alpha * alpha;

  QuantumSpdeBoseUpdateCoefficients coeffs;
  coeffs.inv_omega2 = 1.0 / omega2;
  coeffs.eta_scale = static_cast<jams::Real>(sqrt(2.0 * gamma / h));

  if (discriminant > 0.0) {
    const double beta = sqrt(discriminant);
    const double c = cos(beta * h);
    const double s = sin(beta * h);
    const double inv_beta = 1.0 / beta;

    coeffs.m00 = decay * (c + alpha * inv_beta * s);
    coeffs.m01 = decay * inv_beta * s;
    coeffs.m10 = -decay * omega2 * inv_beta * s;
    coeffs.m11 = decay * (c - alpha * inv_beta * s);
  } else if (discriminant < 0.0) {
    const double beta = sqrt(-discriminant);
    const double c = cosh(beta * h);
    const double s = sinh(beta * h);
    const double inv_beta = 1.0 / beta;

    coeffs.m00 = decay * (c + alpha * inv_beta * s);
    coeffs.m01 = decay * inv_beta * s;
    coeffs.m10 = -decay * omega2 * inv_beta * s;
    coeffs.m11 = decay * (c - alpha * inv_beta * s);
  } else {
    coeffs.m00 = decay * (1.0 + alpha * h);
    coeffs.m01 = decay * h;
    coeffs.m10 = -decay * alpha * alpha * h;
    coeffs.m11 = decay * (1.0 - alpha * h);
  }

  coeffs.force0 = 1.0 - coeffs.m00;
  coeffs.force1 = -coeffs.m10;
  return coeffs;
}

QuantumSpdeZeroPointUpdateCoefficients quantum_spde_zero_point_update_coefficients(
    const double delta_tau, const double omega_max) {
  const double h_omega_max = (kHBarIU * omega_max * delta_tau) / kBoltzmannIU;
  if (h_omega_max <= 0.0) {
    return {};
  }

  constexpr double kWeights[4] = {1.043576, 0.177222, 0.050319, 0.010241};
  constexpr double kLambdaFactors[4] = {1.763817, 0.394613, 0.103506, 0.015873};

  QuantumSpdeZeroPointUpdateCoefficients coeffs;
  coeffs.zero_point_scale = static_cast<jams::Real>((kHBarIU * omega_max) / kBoltzmannIU);
  for (auto i = 0; i < 4; ++i) {
    const double lambda_h = kLambdaFactors[i] * h_omega_max;
    coeffs.decay[i] = exp(-lambda_h);
    coeffs.eta_scale[i] = static_cast<jams::Real>(sqrt(2.0 / lambda_h));
    coeffs.weight[i] = static_cast<jams::Real>(kWeights[i]);
  }
  return coeffs;
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
  const auto seed = static_cast<std::uint64_t>(jams::instance().random_generator()());
  CHECK_CURAND_STATUS(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND_STATUS(curandSetPseudoRandomGeneratorSeed(curand_generator_, seed));
  CHECK_CURAND_STATUS(curandSetStream(curand_generator_, curand_stream_.get()));
  CHECK_CURAND_STATUS(curandGenerateSeeds(curand_generator_));

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
    zero_point_coefficients_ = quantum_spde_zero_point_update_coefficients(delta_tau_, omega_max_);
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

  if (curand_generator_ != nullptr) {
    curandDestroyGenerator(curand_generator_);
    curand_generator_ = nullptr;
  }
}

void CudaQuantumSpdeNoiseGenerator::generate_random_buffers() {
  if (zero_point_) {
    generate_normal(curand_generator_, eta0a_);
    generate_normal(curand_generator_, eta0b_);
  }
  generate_normal(curand_generator_, eta1a_);
  generate_normal(curand_generator_, eta1b_);

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

void CudaQuantumSpdeNoiseGenerator::prepare_fixed_temperature_coefficients(
    const jams::Real temperature) {
  const double reduced_delta_tau = delta_tau_ * static_cast<double>(temperature);
  if (reduced_delta_tau <= 0.0) {
    fast_coefficients_valid_ = false;
    fast_temperature_ = temperature;
    return;
  }

  fast_factor5_ = quantum_spde_bose_update_coefficients(5.0142, 2.7189, reduced_delta_tau);
  fast_factor6_ = quantum_spde_bose_update_coefficients(3.2974, 1.2223, reduced_delta_tau);
  fast_temperature_ = temperature;
  fast_coefficients_valid_ = true;
}

void CudaQuantumSpdeNoiseGenerator::prepare_temperature_profile_coefficients(
    const jams::MultiArray<jams::Real, 1>& temperature) {
  if (static_cast<int>(temperature.size()) != process_count_) {
    throw std::runtime_error("quantum SPDE temperature profile size does not match process count");
  }

  profile_factor5_.resize(process_count_);
  profile_factor6_.resize(process_count_);
  profile_stationary_factor5_.resize(process_count_);
  profile_stationary_factor6_.resize(process_count_);
  profile_has_positive_temperature_ = false;

  const auto* temperature_host = temperature.host_data();
  auto factor5 = profile_factor5_.mutable_host_view();
  auto factor6 = profile_factor6_.mutable_host_view();
  auto stationary5 = profile_stationary_factor5_.mutable_host_view();
  auto stationary6 = profile_stationary_factor6_.mutable_host_view();

  for (auto x = 0; x < process_count_; ++x) {
    const double reduced_delta_tau =
        delta_tau_ * static_cast<double>(temperature_host[x]);
    if (reduced_delta_tau <= 0.0) {
      factor5(x) = {};
      factor6(x) = {};
      stationary5(x) = {};
      stationary6(x) = {};
      continue;
    }

    profile_has_positive_temperature_ = true;
    factor5(x) = quantum_spde_bose_update_coefficients(
        5.0142, 2.7189, reduced_delta_tau);
    factor6(x) = quantum_spde_bose_update_coefficients(
        3.2974, 1.2223, reduced_delta_tau);
    stationary5(x) = quantum_spde_stationary_bose_cholesky(
        5.0142, 2.7189, reduced_delta_tau);
    stationary6(x) = quantum_spde_stationary_bose_cholesky(
        3.2974, 1.2223, reduced_delta_tau);
  }
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
  prepare_fixed_temperature_coefficients(temperature);
}

void CudaQuantumSpdeNoiseGenerator::initialize(
    const Initialization initialization,
    const jams::MultiArray<jams::Real, 1>& temperature) {
  prepare_temperature_profile_coefficients(temperature);
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

  prepare_fixed_temperature_coefficients(temperature);
}

void CudaQuantumSpdeNoiseGenerator::initialize_stationary(
    const jams::MultiArray<jams::Real, 1>& temperature) {
  prepare_temperature_profile_coefficients(temperature);
  zero_state();

  int block_size = 128;
  int grid_size = (process_count_ + block_size - 1) / block_size;
  bool consumed_random_numbers = false;

  cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);

  if (profile_has_positive_temperature_) {
    cuda_thermostat_quantum_spde_stationary_no_zero_profile_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
        zeta6p_.mutable_device_data(), eta1a_.device_data(), eta1b_.device_data(),
        profile_stationary_factor5_.device_data(), profile_stationary_factor6_.device_data(),
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

void CudaQuantumSpdeNoiseGenerator::warmup(
    const unsigned steps,
    const jams::MultiArray<jams::Real, 1>& temperature,
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

  if (temperature == 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise, 0, process_count_ * sizeof(jams::Real), update_stream_.get()));

    if (zero_point_) {
      swap(eta0a_, eta0b_);
      std::swap(eta0a_reusable_, eta0b_reusable_);

      cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
      cudaStreamWaitEvent(curand_stream_.get(), eta0a_reusable_, 0);
      cuda_thermostat_quantum_spde_zero_point_fast_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
          noise, zeta0_.mutable_device_data(), eta0b_.device_data(), sigma, zero_point_coefficients_,
          process_count_);
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
      cudaEventRecord(eta0b_reusable_, update_stream_.get());
      DEBUG_CHECK_CUDA_ASYNC_STATUS

      generate_normal(curand_generator_, eta0a_);

      cudaEventRecord(curand_done_, curand_stream_.get());
      DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

    return;
  }

  const double reduced_delta_tau = delta_tau_ * temperature;
  const bool use_fast_no_zero = fast_coefficients_valid_ && temperature == fast_temperature_;

  // The refill target was used by an earlier update before the swap. Each
  // buffer carries its own reusable event, so CURAND only waits for the buffer
  // it is about to overwrite.
  swap(eta1a_, eta1b_);
  std::swap(eta1a_reusable_, eta1b_reusable_);

  cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
  cudaStreamWaitEvent(curand_stream_.get(), eta1a_reusable_, 0);
  if (use_fast_no_zero) {
    cuda_thermostat_quantum_spde_no_zero_fast_kernel<<<grid_size, block_size, 0, update_stream_.get() >>> (
      noise, zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
      zeta6p_.mutable_device_data(), eta1b_.device_data(), sigma, fast_factor5_, fast_factor6_,
      temperature, process_count_);
  } else {
    cuda_thermostat_quantum_spde_no_zero_kernel<<<grid_size, block_size, 0, update_stream_.get() >>> (
      noise, zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
      zeta6p_.mutable_device_data(), eta1b_.device_data(), sigma, reduced_delta_tau,
      temperature, process_count_);
    prepare_fixed_temperature_coefficients(temperature);
  }
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
  cudaEventRecord(eta1b_reusable_, update_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  generate_normal(curand_generator_, eta1a_);

  if (zero_point_) {
    swap(eta0a_, eta0b_);
    std::swap(eta0a_reusable_, eta0b_reusable_);
    cudaStreamWaitEvent(curand_stream_.get(), eta0a_reusable_, 0);

    cuda_thermostat_quantum_spde_zero_point_fast_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        noise, zeta0_.mutable_device_data(), eta0b_.device_data(), sigma, zero_point_coefficients_,
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    cudaEventRecord(eta0b_reusable_, update_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS

    generate_normal(curand_generator_, eta0a_);
  }

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

void CudaQuantumSpdeNoiseGenerator::update(
    jams::Real* noise,
    const jams::Real* sigma,
    const jams::MultiArray<jams::Real, 1>& temperature) {
  int block_size = 128;
  int grid_size = (process_count_ + block_size - 1) / block_size;

  if (!profile_has_positive_temperature_) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise, 0, process_count_ * sizeof(jams::Real), update_stream_.get()));
  } else {
    swap(eta1a_, eta1b_);
    std::swap(eta1a_reusable_, eta1b_reusable_);

    cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
    cudaStreamWaitEvent(curand_stream_.get(), eta1a_reusable_, 0);
    cuda_thermostat_quantum_spde_no_zero_profile_fast_kernel<<<grid_size, block_size, 0, update_stream_.get() >>> (
      noise, zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
      zeta6p_.mutable_device_data(), eta1b_.device_data(), sigma, profile_factor5_.device_data(),
      profile_factor6_.device_data(), temperature.device_data(), process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    cudaEventRecord(eta1b_reusable_, update_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS

    generate_normal(curand_generator_, eta1a_);
  }

  if (zero_point_) {
    swap(eta0a_, eta0b_);
    std::swap(eta0a_reusable_, eta0b_reusable_);
    cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
    cudaStreamWaitEvent(curand_stream_.get(), eta0a_reusable_, 0);

    cuda_thermostat_quantum_spde_zero_point_fast_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        noise, zeta0_.mutable_device_data(), eta0b_.device_data(), sigma, zero_point_coefficients_,
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    cudaEventRecord(eta0b_reusable_, update_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS

    generate_normal(curand_generator_, eta0a_);
  }

  if (profile_has_positive_temperature_ || zero_point_) {
    cudaEventRecord(curand_done_, curand_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
}

void CudaQuantumSpdeNoiseGenerator::synchronize() {
  update_stream_.synchronize();
  curand_stream_.synchronize();
}

}  // namespace jams
