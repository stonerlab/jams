// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cuda_llg_heun.h"

#include <iostream>
#include <string>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/cuda/cuda_common.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/helpers/defaults.h"
#include "jams/interface/config.h"
#include "jams/solvers/llg_rhs.h"

namespace {

__device__ void heun_rhs(const double spin[3], const jams::Real* field, const jams::Real* noise,
                         const jams::Real inv_mu, const jams::Real gyro_eff, const jams::Real alpha,
                         double ds_dt[3]) {
  double effective_field[3];
  for (int n = 0; n < 3; ++n) {
    const jams::Real value = field[n] * inv_mu + noise[n];
    effective_field[n] = value;
  }
  jams::solvers::llg_rhs(spin, effective_field, gyro_eff, alpha, ds_dt);
}

__device__ void store_normalised_spin(const double spin[3], double* destination) {
  const double norm_sq = norm_squared(spin);
  const double inv_norm = norm_sq > 0.0 ? rsqrt(norm_sq) : 0.0;
  for (int n = 0; n < 3; ++n) {
    destination[n] = spin[n] * inv_norm;
  }
}

__global__ void heun_stage_1(double* spins, double* ds_dt, double* initial_spins,
                             const jams::Real* fields, const jams::Real* noise,
                             const jams::Real* gyro_eff, const jams::Real* inverse_moments,
                             const jams::Real* damping, const double dt,
                             const unsigned int num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;
  const unsigned int offset = 3 * spin_index;

  const double spin[3] = {spins[offset], spins[offset + 1], spins[offset + 2]};
  double k1[3];
  heun_rhs(spin, fields + offset, noise + offset, inverse_moments[spin_index], gyro_eff[spin_index],
           damping[spin_index], k1);

  double stage_2_spin[3];
  for (int n = 0; n < 3; ++n) {
    initial_spins[offset + n] = spin[n];
    ds_dt[offset + n] = 0.5 * k1[n];
    stage_2_spin[n] = spin[n] + dt * k1[n];
  }
  store_normalised_spin(stage_2_spin, spins + offset);
}

__global__ void heun_stage_2(double* spins, double* ds_dt, const double* initial_spins,
                             const jams::Real* fields, const jams::Real* noise,
                             const jams::Real* gyro_eff, const jams::Real* inverse_moments,
                             const jams::Real* damping, const double dt,
                             const unsigned int num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;
  const unsigned int offset = 3 * spin_index;

  const double spin[3] = {spins[offset], spins[offset + 1], spins[offset + 2]};
  double k2[3];
  heun_rhs(spin, fields + offset, noise + offset, inverse_moments[spin_index], gyro_eff[spin_index],
           damping[spin_index], k2);

  double final_spin[3];
  for (int n = 0; n < 3; ++n) {
    const unsigned int index = offset + n;
    // Stage 1 stored k1 / 2. Complete the average with k2 / 2.
    ds_dt[index] += 0.5 * k2[n];
    final_spin[n] = initial_spins[index] + dt * ds_dt[index];
  }
  store_normalised_spin(final_spin, spins + offset);
}

}  // namespace

void CUDAHeunLLGSolver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);
  initialize_gyro_eff(settings, gyro_eff_);

  const auto thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(
      thermostat_name, time_step(), jams::thermal_field_coupling(gyro_eff_)));
  std::cout << "  thermostat " << thermostat_name << "\n";

  spins_initial_.resize(globals::num_spins, 3);
}

void CUDAHeunLLGSolver::run() {
  const double step_start_time = time_;
  const double dt = step_size_;
  const auto stream = jams::instance().cuda_master_stream().get();
  const dim3 block_size = {256, 1, 1};
  const auto num_spins = static_cast<unsigned int>(globals::num_spins);
  const auto grid_size = cuda_grid_size(block_size, {num_spins, 1, 1});

  update_thermostat_on_stream(stream);
  const auto* noise = thermostat_->device_data();

  compute_fields();
  heun_stage_1<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), globals::ds_dt.mutable_device_data(),
      spins_initial_.mutable_device_data(), globals::h.device_data(), noise,
      gyro_eff_.device_data(), globals::inv_mus.device_data(), globals::alpha.device_data(), dt,
      num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  time_ = step_start_time + dt;
  compute_fields();
  heun_stage_2<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), globals::ds_dt.mutable_device_data(),
      spins_initial_.device_data(), globals::h.device_data(), noise, gyro_eff_.device_data(),
      globals::inv_mus.device_data(), globals::alpha.device_data(), dt, num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);
  record_spin_barrier_event();

  advance_time();
}
