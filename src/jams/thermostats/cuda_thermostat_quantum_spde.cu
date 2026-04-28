// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <random>
#include <mutex>
#include <algorithm>

#include <jams/common.h>
#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_array_kernels.h"

#include "jams/thermostats/cuda_thermostat_quantum_spde.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde_kernel.cuh"

#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_array_kernels.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_common.h"
#include "jams/monitors/magnetisation.h"
#include <jams/helpers/exception.h>

namespace {

struct Cholesky2 {
  double l00 = 0.0;
  double l10 = 0.0;
  double l11 = 0.0;
};

void generate_normal(jams::MultiArray<jams::Real, 1>& data) {
#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(
      jams::instance().curand_generator(), data.mutable_device_data(), data.size(), 0.0, 1.0));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(
      jams::instance().curand_generator(), data.mutable_device_data(), data.size(), 0.0, 1.0));
#endif
}

void bose_exact_update_host(const double gamma, const double omega, const double eta0,
                            const double h, double z[2]) {
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

Cholesky2 stationary_bose_cholesky(const double gamma, const double omega, const double h) {
  if (h <= 0.0) {
    return {};
  }

  double z[2] = {1.0, 0.0};
  bose_exact_update_host(gamma, omega, 0.0, h, z);
  const double a00 = z[0];
  const double a10 = z[1];

  z[0] = 0.0;
  z[1] = 1.0;
  bose_exact_update_host(gamma, omega, 0.0, h, z);
  const double a01 = z[0];
  const double a11 = z[1];

  z[0] = 0.0;
  z[1] = 0.0;
  bose_exact_update_host(gamma, omega, 1.0, h, z);
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

  Cholesky2 factor;
  factor.l00 = sqrt(p00);
  factor.l10 = p01 / factor.l00;
  factor.l11 = sqrt(std::max(0.0, p11 - factor.l10 * factor.l10));
  return factor;
}

}  // namespace

CudaThermostatQuantumSpde::CudaThermostatQuantumSpde(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins)
: Thermostat(temperature, sigma, timestep, num_spins)
  {
   std::cout << "\n  initialising quantum-spde-gpu thermostat\n";

  cudaEventCreateWithFlags(&curand_done_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_stream_ = CudaStream(CudaStream::Priority::LOW);
  const libconfig::Setting* thermostat_settings = nullptr;
  if (globals::config->exists("thermostat")) {
    thermostat_settings = &globals::config->lookup("thermostat");
  }

  zeta5_.resize(num_spins * 3).zero();
  zeta5p_.resize(num_spins * 3).zero();
  zeta6_.resize(num_spins * 3).zero();
  zeta6p_.resize(num_spins * 3).zero();
  eta1a_.resize(2 * num_spins * 3).zero();
  eta1b_.resize(2 * num_spins * 3).zero();

   do_zero_point_ = thermostat_settings != nullptr
       && jams::config_optional<bool>(*thermostat_settings, "zero_point", false);
   if (do_zero_point_) {
     zeta0_.resize(4 * num_spins * 3).zero();
     eta0a_.resize(4 * num_spins * 3).zero();
     eta0b_.resize(4 * num_spins * 3).zero();
   }

   double t_warmup = 1e-10; // 0.1 ns
   if (thermostat_settings != nullptr) {
     t_warmup = jams::config_optional<double>(*thermostat_settings, "warmup_time", t_warmup);
   }
   t_warmup = t_warmup / 1e-12; // convert to ps
   const bool do_warmup = thermostat_settings != nullptr
       && jams::config_optional<bool>(*thermostat_settings, "warmup", false);
   const auto initialization = lowercase(thermostat_settings != nullptr
       ? jams::config_optional<std::string>(*thermostat_settings, "initialization", "stationary")
       : std::string("stationary"));
   if (initialization != "stationary" && initialization != "zero") {
     throw jams::ConfigException(
         *thermostat_settings, "initialization must be either 'stationary' or 'zero'");
   }

   omega_max_ = 25.0 * kTwoPi;
   if (thermostat_settings != nullptr) {
     omega_max_ = jams::config_optional<double>(*thermostat_settings, "w_max", omega_max_);
   }

   double dt_thermostat = timestep;
   delta_tau_ = (dt_thermostat * kBoltzmannIU) / kHBarIU;

   std::cout << "    omega_max (THz) " << omega_max_ / (kTwoPi) << "\n";
   std::cout << "    hbar*w/kB " << (kHBarIU * omega_max_) / (kBoltzmannIU) << "\n";
   std::cout << "    t_step " << dt_thermostat << "\n";
   std::cout << "    delta tau " << delta_tau_ << "\n";
   std::cout << "    initialization " << initialization << "\n";
   std::cout << "    warmup " << std::boolalpha << do_warmup << "\n";

  generate_random_buffers();

    bool use_gilbert_prefactor = jams::config_optional<bool>(
        globals::config->lookup("solver"), "gilbert_prefactor", false);
    std::cout << "    llg gilbert_prefactor " << use_gilbert_prefactor << "\n";

    for (int i = 0; i < num_spins; ++i) {
      for (int j = 0; j < 3; ++j) {
        double denominator = 1.0;
        if (use_gilbert_prefactor) {
          denominator = 1.0 + pow2(globals::alpha(i));
        }
        sigma_(i,j) = static_cast<jams::Real>((kBoltzmannIU) * sqrt((2.0 * globals::alpha(i))
                                            / (kHBarIU * globals::gyro(i) * globals::mus(i) * denominator)));
      }
    }

  if (initialization == "stationary") {
    initialize_stationary();
  }

  auto num_warm_up_steps = static_cast<unsigned>(t_warmup / dt_thermostat);
  if (do_warmup && num_warm_up_steps > 0) {
    warmup(num_warm_up_steps);
  }
}

void CudaThermostatQuantumSpde::generate_random_buffers() {
  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
  if (do_zero_point_) {
    generate_normal(eta0a_);
    generate_normal(eta0b_);
  }
  generate_normal(eta1a_);
  generate_normal(eta1b_);

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

void CudaThermostatQuantumSpde::initialize_stationary() {
  std::cout << "initialising thermostat from stationary distribution @ ";
  std::cout << this->temperature() << "K" << std::endl;

  int block_size = 128;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;
  bool consumed_random_numbers = false;

  cudaStreamWaitEvent(cuda_stream_.get(), curand_done_, 0);

  const double reduced_delta_tau = delta_tau_ * this->temperature();
  if (reduced_delta_tau > 0.0) {
    const auto factor5 = stationary_bose_cholesky(5.0142, 2.7189, reduced_delta_tau);
    const auto factor6 = stationary_bose_cholesky(3.2974, 1.2223, reduced_delta_tau);
    cuda_thermostat_quantum_spde_stationary_no_zero_kernel <<< grid_size, block_size, 0, cuda_stream_.get() >>> (
        zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
        zeta6p_.mutable_device_data(), eta1a_.device_data(), eta1b_.device_data(),
        factor5.l00, factor5.l10, factor5.l11, factor6.l00, factor6.l10, factor6.l11,
        globals::num_spins3);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    consumed_random_numbers = true;
  }

  if (do_zero_point_) {
    const double zero_point_delta_tau = (kHBarIU * omega_max_ * delta_tau_) / kBoltzmannIU;
    cuda_thermostat_quantum_spde_stationary_zero_point_kernel <<< grid_size, block_size, 0, cuda_stream_.get() >>> (
        zeta0_.mutable_device_data(), eta0a_.device_data(), static_cast<jams::Real>(zero_point_delta_tau),
        globals::num_spins3);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    consumed_random_numbers = true;
  }

  if (consumed_random_numbers) {
    CHECK_CUDA_STATUS(cudaStreamSynchronize(cuda_stream_.get()));
    generate_random_buffers();
  }
}

void CudaThermostatQuantumSpde::warmup(unsigned steps) {
  std::cout << "warming up thermostat " << steps << " steps @ ";
  std::cout << this->temperature() << "K" << std::endl;

  for (auto i = 0u; i < steps; ++i) {
    CudaThermostatQuantumSpde::update();
  }
}

void CudaThermostatQuantumSpde::update() {
  int block_size = 128;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  const jams::Real temperature = this->temperature();
  const double zero_point_delta_tau = (kHBarIU * omega_max_ * delta_tau_) / kBoltzmannIU;
  const double zero_point_scale = (kHBarIU * omega_max_) / kBoltzmannIU;

  if (temperature == 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise_.mutable_device_data(), 0, noise_.bytes(), cuda_stream_.get()));

    if (do_zero_point_) {
      swap(eta0a_, eta0b_);

      cudaStreamWaitEvent(cuda_stream_.get(), curand_done_, 0);
      cuda_thermostat_quantum_spde_zero_point_kernel <<< grid_size, block_size, 0, cuda_stream_.get() >>> (
          noise_.mutable_device_data(), zeta0_.mutable_device_data(), eta0b_.device_data(), sigma_.device_data(),
          static_cast<jams::Real>(zero_point_delta_tau), static_cast<jams::Real>(zero_point_scale),
          globals::num_spins3);
      DEBUG_CHECK_CUDA_ASYNC_STATUS;

      CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
      generate_normal(eta0a_);

      cudaEventRecord(curand_done_, curand_stream_.get());
      DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

    return;
  }

  const double reduced_delta_tau = delta_tau_ * temperature;

  // We swap pointers to arrays of random numbers and then generate new numbers AFTER the kernel invocation. This
  // allows the expensive generation to be hidden by multiplexing with other streams all the way until we next get back
  // to the thermostat update.
  swap(eta1a_, eta1b_);

  cudaStreamWaitEvent(cuda_stream_.get(), curand_done_, 0);
  cuda_thermostat_quantum_spde_no_zero_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>> (
    noise_.mutable_device_data(), zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(), zeta6p_.mutable_device_data(),
    eta1b_.device_data(), sigma_.device_data(), reduced_delta_tau, temperature, globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
  generate_normal(eta1a_);

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS


  if (do_zero_point_) {
    swap(eta0a_, eta0b_);

    cuda_thermostat_quantum_spde_zero_point_kernel <<< grid_size, block_size, 0, cuda_stream_.get() >>> (
        noise_.mutable_device_data(), zeta0_.mutable_device_data(), eta0b_.device_data(), sigma_.device_data(),
        static_cast<jams::Real>(zero_point_delta_tau), static_cast<jams::Real>(zero_point_scale),
        globals::num_spins3);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;

    CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
    generate_normal(eta0a_);

    cudaEventRecord(curand_done_, curand_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
}
