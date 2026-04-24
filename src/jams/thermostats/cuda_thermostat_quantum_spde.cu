// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/cuda_thermostat_quantum_spde.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include <jams/common.h>

#include "jams/cuda/cuda_common.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/maths.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde_kernel.cuh"

namespace {

void bose_exact_update_host(const double gamma,
                            const double omega,
                            const double eta0,
                            const double h,
                            double z[2]) {
  const double omega2 = omega * omega;
  const double alpha = 0.5 * gamma;
  const double decay = std::exp(-alpha * h);
  const double force_eq = eta0 / omega2;

  const double y0 = z[0] - force_eq;
  const double v0 = z[1];

  const double discriminant = omega2 - alpha * alpha;
  if (discriminant > 0.0) {
    const double beta = std::sqrt(discriminant);
    const double c = std::cos(beta * h);
    const double s = std::sin(beta * h);
    const double inv_beta = 1.0 / beta;

    const double y1 = decay * (y0 * c + (v0 + alpha * y0) * inv_beta * s);
    const double v1 = decay * (v0 * c - (alpha * v0 + omega2 * y0) * inv_beta * s);

    z[0] = y1 + force_eq;
    z[1] = v1;
    return;
  }

  if (discriminant < 0.0) {
    const double beta = std::sqrt(-discriminant);
    const double c = std::cosh(beta * h);
    const double s = std::sinh(beta * h);
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

std::array<double, 3> solve_linear_system_3x3(
    const std::array<std::array<double, 4>, 3>& augmented) {
  auto a = augmented;
  for (int pivot = 0; pivot < 3; ++pivot) {
    int pivot_row = pivot;
    for (int row = pivot + 1; row < 3; ++row) {
      if (std::abs(a[row][pivot]) > std::abs(a[pivot_row][pivot])) {
        pivot_row = row;
      }
    }

    std::swap(a[pivot], a[pivot_row]);
    const double inv_pivot = 1.0 / a[pivot][pivot];
    for (int col = pivot; col < 4; ++col) {
      a[pivot][col] *= inv_pivot;
    }

    for (int row = 0; row < 3; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = a[row][pivot];
      for (int col = pivot; col < 4; ++col) {
        a[row][col] -= factor * a[pivot][col];
      }
    }
  }

  return {a[0][3], a[1][3], a[2][3]};
}

std::array<double, 3> stationary_oscillator_covariance(const double gamma,
                                                       const double omega,
                                                       const double h) {
  const double eta_scale = std::sqrt(2.0 * gamma / h);

  double basis0[2] = {1.0, 0.0};
  bose_exact_update_host(gamma, omega, 0.0, h, basis0);
  double basis1[2] = {0.0, 1.0};
  bose_exact_update_host(gamma, omega, 0.0, h, basis1);
  double noise[2] = {0.0, 0.0};
  bose_exact_update_host(gamma, omega, eta_scale, h, noise);

  const double a = basis0[0];
  const double b = basis1[0];
  const double c = basis0[1];
  const double d = basis1[1];

  const double q00 = noise[0] * noise[0];
  const double q01 = noise[0] * noise[1];
  const double q11 = noise[1] * noise[1];

  const std::array<std::array<double, 4>, 3> system = {{
      {{1.0 - a * a, -2.0 * a * b, -b * b, q00}},
      {{-a * c, 1.0 - a * d - b * c, -b * d, q01}},
      {{-c * c, -2.0 * c * d, 1.0 - d * d, q11}},
  }};

  return solve_linear_system_3x3(system);
}

double stationary_first_coordinate_variance(const double gamma,
                                            const double omega,
                                            const double h) {
  return stationary_oscillator_covariance(gamma, omega, h)[0];
}

double stationary_ou_variance(const double lambda, const double h) {
  const double x = lambda * h;
  if (x == 0.0) {
    return 1.0;
  }
  return (2.0 * std::tanh(0.5 * x)) / x;
}

std::array<double, 2> sample_bivariate_gaussian(
    const std::array<double, 3>& covariance,
    std::normal_distribution<double>& normal_distribution,
    jams::RandomGeneratorType& random_generator) {
  const double variance0 = std::max(0.0, covariance[0]);
  const double covariance01 = covariance[1];
  const double variance1 = std::max(0.0, covariance[2]);

  const double xi0 = normal_distribution(random_generator);
  const double xi1 = normal_distribution(random_generator);
  const double stddev0 = std::sqrt(variance0);
  const double conditional_mean_factor = stddev0 > 0.0 ? covariance01 / stddev0 : 0.0;
  const double conditional_variance =
      std::max(0.0, variance1 - covariance01 * covariance01 / std::max(variance0, 1.0e-30));

  return {
      stddev0 * xi0,
      conditional_mean_factor * xi0 + std::sqrt(conditional_variance) * xi1,
  };
}

}  // namespace

CudaQuantumSpdeNoiseGenerator::CudaQuantumSpdeNoiseGenerator(
    const jams::Real& temperature,
    const jams::Real timestep,
    const int num_vectors,
    const CudaQuantumSpdeNoiseGeneratorConfig& config)
    : NoiseGenerator(temperature, num_vectors),
      do_zero_point_(config.zero_point),
      delta_tau_((timestep * kBoltzmannIU) / kHBarIU),
      omega_max_(config.omega_max),
      stationary_temperature_(temperature),
      num_channels_(3 * num_vectors) {
  std::cout << "\n  initialising quantum-spde-gpu noise generator\n";

  cudaEventCreateWithFlags(&curand_done_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_stream_ = CudaStream(CudaStream::Priority::LOW);

  zero(zeta5_.resize(num_channels_));
  zero(zeta5p_.resize(num_channels_));
  zero(zeta6_.resize(num_channels_));
  zero(zeta6p_.resize(num_channels_));
  zero(eta1a_.resize(2 * num_channels_));
  zero(eta1b_.resize(2 * num_channels_));

  if (do_zero_point_) {
    zero(zeta0_.resize(4 * num_channels_));
    zero(eta0a_.resize(4 * num_channels_));
    zero(eta0b_.resize(4 * num_channels_));
  }

  std::cout << "    omega_max (THz) " << omega_max_ / kTwoPi << "\n";
  std::cout << "    hbar*w/kB " << (kHBarIU * omega_max_) / kBoltzmannIU << "\n";
  std::cout << "    t_step " << timestep << "\n";
  std::cout << "    delta tau " << delta_tau_ << "\n";

  initialize_stationary_state();

  CHECK_CURAND_STATUS(
      curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));

  if (do_zero_point_) {
#ifdef DO_MIXED_PRECISION
    CHECK_CURAND_STATUS(curandGenerateNormal(
        jams::instance().curand_generator(), eta0a_.device_data(), eta0a_.size(), 0.0f, 1.0f));
    CHECK_CURAND_STATUS(curandGenerateNormal(
        jams::instance().curand_generator(), eta0b_.device_data(), eta0b_.size(), 0.0f, 1.0f));
#else
    CHECK_CURAND_STATUS(curandGenerateNormalDouble(
        jams::instance().curand_generator(), eta0a_.device_data(), eta0a_.size(), 0.0, 1.0));
    CHECK_CURAND_STATUS(curandGenerateNormalDouble(
        jams::instance().curand_generator(), eta0b_.device_data(), eta0b_.size(), 0.0, 1.0));
#endif
  }

#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(
      jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0f, 1.0f));
  CHECK_CURAND_STATUS(curandGenerateNormal(
      jams::instance().curand_generator(), eta1b_.device_data(), eta1b_.size(), 0.0f, 1.0f));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(
      jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(
      jams::instance().curand_generator(), eta1b_.device_data(), eta1b_.size(), 0.0, 1.0));
#endif

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

CudaQuantumSpdeNoiseGenerator::~CudaQuantumSpdeNoiseGenerator() {
  if (curand_done_ != nullptr) {
    cudaEventDestroy(curand_done_);
    curand_done_ = nullptr;
  }
}

void CudaQuantumSpdeNoiseGenerator::initialize_stationary_state() {
  if (this->temperature() == 0) {
    return;
  }

  const double reduced_delta_tau = delta_tau_ * this->temperature();
  const double reduced_omega_max =
      (kHBarIU * omega_max_) / (kBoltzmannIU * this->temperature());

  auto& random_generator = jams::instance().random_generator();
  std::normal_distribution<double> normal_distribution(0.0, 1.0);

  const auto initialize_oscillator_pair =
      [&](jams::MultiArray<double, 1>& zeta,
          jams::MultiArray<double, 1>& zeta_prime,
          const double gamma,
          const double omega) {
        const auto covariance =
            stationary_oscillator_covariance(gamma, omega, reduced_delta_tau);
        for (int i = 0; i < num_channels_; ++i) {
          const auto sample =
              sample_bivariate_gaussian(covariance, normal_distribution, random_generator);
          zeta(i) = sample[0];
          zeta_prime(i) = sample[1];
        }
      };

  initialize_oscillator_pair(zeta5_, zeta5p_, 5.0142, 2.7189);
  initialize_oscillator_pair(zeta6_, zeta6p_, 3.2974, 1.2223);

  if (!do_zero_point_) {
    return;
  }

  constexpr std::array<double, 4> kLambdaScale = {
      1.763817,
      0.394613,
      0.103506,
      0.015873,
  };

  for (int channel = 0; channel < num_channels_; ++channel) {
    for (int mode = 0; mode < static_cast<int>(kLambdaScale.size()); ++mode) {
      const double lambda = kLambdaScale[mode] * reduced_omega_max;
      const double variance = stationary_ou_variance(lambda, reduced_delta_tau);
      zeta0_(4 * channel + mode) = std::sqrt(variance) * normal_distribution(random_generator);
    }
  }
}

void CudaQuantumSpdeNoiseGenerator::update() {
  if (this->temperature() == 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise_.device_data(), 0, noise_.bytes(), cuda_stream_.get()));
    return;
  }

  if (stationary_temperature_ == jams::Real(0.0)) {
    stationary_temperature_ = this->temperature();
  }

  if (this->temperature() != stationary_temperature_) {
    throw std::runtime_error(
        "CudaQuantumSpdeNoiseGenerator does not support nonzero dynamic temperature changes");
  }

  const int block_size = 128;
  const int grid_size = (num_channels_ + block_size - 1) / block_size;

  const double reduced_omega_max =
      (kHBarIU * omega_max_) / (kBoltzmannIU * this->temperature());
  const double reduced_delta_tau = delta_tau_ * this->temperature();
  const jams::Real temperature = this->temperature();

  swap(eta1a_, eta1b_);
  if (do_zero_point_) {
    swap(eta0a_, eta0b_);
  }

  cudaStreamWaitEvent(cuda_stream_.get(), curand_done_, 0);
  cuda_thermostat_quantum_spde_no_zero_kernel<<<grid_size, block_size, 0, cuda_stream_.get()>>>(
      noise_.device_data(),
      zeta5_.device_data(),
      zeta5p_.device_data(),
      zeta6_.device_data(),
      zeta6p_.device_data(),
      eta1b_.device_data(),
      reduced_delta_tau,
      temperature,
      num_channels_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(
      jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0f, 1.0f));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(
      jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));
#endif

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  if (do_zero_point_) {
    cuda_thermostat_quantum_spde_zero_point_kernel<<<grid_size, block_size, 0, cuda_stream_.get()>>>(
        noise_.device_data(),
        zeta0_.device_data(),
        eta0b_.device_data(),
        reduced_delta_tau,
        temperature,
        reduced_omega_max,
        num_channels_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;

    CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
#ifdef DO_MIXED_PRECISION
    CHECK_CURAND_STATUS(curandGenerateNormal(
        jams::instance().curand_generator(), eta0a_.device_data(), eta0a_.size(), 0.0f, 1.0f));
#else
    CHECK_CURAND_STATUS(curandGenerateNormalDouble(
        jams::instance().curand_generator(), eta0a_.device_data(), eta0a_.size(), 0.0, 1.0));
#endif

    cudaEventRecord(curand_done_, curand_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
}

double CudaQuantumSpdeNoiseGenerator::stationary_variance() const {
  if (this->temperature() == 0) {
    return 0.0;
  }
  if (do_zero_point_) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double h = delta_tau_ * this->temperature();
  const double var5 = no_zero_process_variance_component(5.0142, 2.7189, h);
  const double var6 = no_zero_process_variance_component(3.2974, 1.2223, h);
  return pow2(this->temperature())
         * (pow2(1.8315) * var5 + pow2(0.3429) * var6);
}

double CudaQuantumSpdeNoiseGenerator::no_zero_process_variance_component(
    const double gamma,
    const double omega,
    const double step) {
  return stationary_first_coordinate_variance(gamma, omega, step);
}
