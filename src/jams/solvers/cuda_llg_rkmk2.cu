//
// Created by Joseph Barker on 05/01/2026.
//

#include "jams/solvers/cuda_llg_rkmk2.h"

#include <iostream>
#include <string>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/helpers/defaults.h"
#include "jams/interface/config.h"
#include "jams/solvers/cuda_solver_functions.cuh"

namespace {

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rkmk2_stage_1(const double* stage_spins, double* initial_spins, double* phi_dev,
                              double* output_spins, jams::Real* field_spin_cache,
                              const jams::Real* fields,  // field at the same time as s_step
                              const GyroParam gyro, const FieldScaleParam field_scale,
                              const AlphaParam alpha, const unsigned num_spins, const double dt) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned int offset = 3u * spin_index;

  jams::Real h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = field_scale.scale(fields[offset + n], spin_index);
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = stage_spins[offset + n];
    initial_spins[offset + n] = s[n];
  }

  double omega[3];
  omega_llg(s, h, gyro.get(spin_index), alpha.get(spin_index), omega);

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = omega[n] * dt * 0.5;
  }

  for (auto n = 0; n < 3; ++n) {
    phi_dev[offset + n] = phi[n];
  }

  double s_out[3];
  rodrigues_rotate(phi, s, s_out);
  rkmk_store_spin_and_cache(output_spins, field_spin_cache, offset, s_out);
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rkmk2_stage_2(const double* initial_spins, const double* stage_spins,
                              const double* phi_dev, double* output_spins,
                              jams::Real* field_spin_cache,
                              const jams::Real* fields,  // field at the same time as s_step
                              const jams::Real* noise, const GyroParam gyro,
                              const FieldScaleParam field_scale, const AlphaParam alpha,
                              const unsigned num_spins, const double dt, const double noise_dt) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned int offset = 3u * spin_index;

  jams::Real h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = field_scale.scale(fields[offset + n], spin_index);
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = stage_spins[offset + n];
  }

  double omega[3];
  omega_llg(s, h, gyro.get(spin_index), alpha.get(spin_index), omega);

  double v2[3];
  for (auto n = 0; n < 3; ++n) {
    v2[n] = omega[n] * dt;
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = phi_dev[offset + n];
  }

  double k[3];
  dexp_inv_so3(phi, v2, k);

  double s_init[3];
  for (auto n = 0; n < 3; ++n) {
    s_init[n] = initial_spins[offset + n];
  }

  double s_out[3];
  rodrigues_rotate(k, s_init, s_out);

  double s_noisy[3];
  rkmk_noise_step_rodrigues(s_out, noise, gyro, alpha, spin_index, offset, noise_dt, s_noisy);
  rkmk_store_spin_and_cache(output_spins, field_spin_cache, offset, s_noisy);
}

}  // namespace

void CUDALLGRKMK2Solver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);

  const auto thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", jams::defaults::solver_gpu_thermostat);
  // Strang splitting means the thermostat timestep is 1/2 of the RKMK time step
  initialize_gyro_eff(settings, gyro_eff_);
  register_thermostat(Thermostat::create(
      thermostat_name, 0.5 * time_step(), jams::thermal_field_coupling(gyro_eff_)));

  std::cout << "  thermostat " << thermostat_name << "\n";

  const auto gyro_eff_choice = cuda_spin_parameter_choice(gyro_eff_);
  gyro_eff_is_uniform_ = gyro_eff_choice.is_uniform;
  gyro_eff_uniform_value_ = gyro_eff_choice.uniform_value;
  const auto alpha_choice = cuda_spin_parameter_choice(globals::alpha);
  alpha_is_uniform_ = alpha_choice.is_uniform;
  alpha_uniform_value_ = alpha_choice.uniform_value;
  const auto mus_choice = cuda_field_scale_choice(globals::inv_mus);
  mus_is_uniform_ = mus_choice.is_uniform;
  mus_uniform_inv_value_ = mus_choice.uniform_inv_mus;

  phi_.resize(globals::num_spins, 3);
  spins_initial_.resize(globals::num_spins, 3);
}

void CUDALLGRKMK2Solver::run() {
  dispatch_cuda_spin_parameters(
      {gyro_eff_is_uniform_, gyro_eff_uniform_value_}, {alpha_is_uniform_, alpha_uniform_value_},
      gyro_eff_, globals::alpha, [this](const auto gyro, const auto alpha) {
        dispatch_cuda_field_scale({mus_is_uniform_, mus_uniform_inv_value_}, globals::inv_mus,
                                  [this, gyro, alpha](const auto field_scale) {
                                    run_with_parameters(gyro, alpha, field_scale);
                                  });
      });
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
void CUDALLGRKMK2Solver::run_with_parameters(const GyroParam gyro, const AlphaParam alpha,
                                             const FieldScaleParam field_scale) {
  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;
  const auto stream = jams::instance().cuda_master_stream().get();

  const dim3 block_size = {256, 1, 1};
  const auto grid_size =
      cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  jams::Real* field_spin_cache = mutable_field_spin_cache_device_data();

  update_thermostat_on_stream(stream);

  cuda_llg_noise_step_rodrigues_cache_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), field_spin_cache, thermostat_->device_data(), gyro, alpha,
      globals::num_spins, half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);
  record_spin_and_field_cache_barrier_event();

  compute_fields();

  rkmk2_stage_1<<<grid_size, block_size, 0, stream>>>(
      globals::s.device_data(), spins_initial_.mutable_device_data(), phi_.mutable_device_data(),
      globals::s.mutable_device_data(), field_spin_cache, globals::h.device_data(), gyro,
      field_scale, alpha, globals::num_spins, dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_and_field_cache_barrier_event();

  time_ = step_start_time + half_dt;

  compute_fields();

  update_thermostat_on_stream(stream);

  rkmk2_stage_2<<<grid_size, block_size, 0, stream>>>(
      spins_initial_.device_data(), globals::s.mutable_device_data(), phi_.device_data(),
      globals::s.mutable_device_data(), field_spin_cache, globals::h.device_data(),
      thermostat_->device_data(), gyro, field_scale, alpha, globals::num_spins, dt, half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);
  record_spin_and_field_cache_barrier_event();

  advance_time();
}
