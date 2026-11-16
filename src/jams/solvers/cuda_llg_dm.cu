// Depondt–Mertens with a thermal half-step on each side of the deterministic step.

#include "jams/solvers/cuda_llg_dm.h"

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
__global__ void dm_predict(double* spins, double* initial_spins, double* omega1_dev,
                           const jams::Real* fields,  // field at time t_n for S_n
                           const GyroParam gyro, const FieldScaleParam field_scale,
                           const AlphaParam alpha, const unsigned num_spins, const double dt) {
  const unsigned spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned offset = 3u * spin_index;

  jams::Real h[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    h[n] = field_scale.scale(fields[offset + n], spin_index);
  }

  double s[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    s[n] = spins[offset + n];
    initial_spins[offset + n] = s[n];
  }

  double omega1[3];
  omega_llg(s, h, gyro.get(spin_index), alpha.get(spin_index), omega1);

// store ω_n
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    omega1_dev[offset + n] = omega1[n];
  }

  // predictor: S* = R(Δt ω_n) S_n
  double phi[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    phi[n] = omega1[n] * dt;
  }

  double s_pred[3];
  rodrigues_rotate(phi, s, s_pred);

#pragma unroll
  for (int n = 0; n < 3; ++n) {
    spins[offset + n] = s_pred[n];
  }
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void dm_correct(const double* initial_spins,  // original S_n
                           const double* omega1_dev,     // ω_n
                           double* spins,
                           const jams::Real* fields,  // field at time t_{n+1} (or t_n+Δt) for S*
                           const GyroParam gyro, const FieldScaleParam field_scale,
                           const AlphaParam alpha, const unsigned num_spins, const double dt) {
  const unsigned spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned offset = 3u * spin_index;

  jams::Real h[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    h[n] = field_scale.scale(fields[offset + n], spin_index);
  }

  double s_pred[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    s_pred[n] = spins[offset + n];
  }

  double omega2[3];
  omega_llg(s_pred, h, gyro.get(spin_index), alpha.get(spin_index), omega2);

  double omega1[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    omega1[n] = omega1_dev[offset + n];
  }

  // average angular velocities (DM)
  double omega_bar[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    omega_bar[n] = 0.5 * (omega1[n] + omega2[n]);
  }

  // rotate original spin S_n with averaged ω
  double s_init[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    s_init[n] = initial_spins[offset + n];
  }

  double phi[3];
#pragma unroll
  for (int n = 0; n < 3; ++n) {
    phi[n] = omega_bar[n] * dt;
  }

  double s_out[3];
  rodrigues_rotate(phi, s_init, s_out);

#pragma unroll
  for (int n = 0; n < 3; ++n) {
    spins[offset + n] = s_out[n];
  }
}

}  // namespace

void CUDALLGDMSolver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);

  const auto thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", jams::defaults::solver_gpu_thermostat);

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

  spins_initial_.resize(globals::num_spins, 3);
  omega1_.resize(globals::num_spins, 3);
}

void CUDALLGDMSolver::run() {
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
void CUDALLGDMSolver::run_with_parameters(const GyroParam gyro, const AlphaParam alpha,
                                          const FieldScaleParam field_scale) {
  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;
  const auto stream = jams::instance().cuda_master_stream().get();

  const dim3 block_size = {256, 1, 1};
  const auto grid_size =
      cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  // Thermal half-step
  update_thermostat_on_stream(stream);

  cuda_llg_noise_step_rodrigues_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), thermostat_->device_data(), gyro, alpha, globals::num_spins,
      half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);
  record_spin_barrier_event();

  compute_fields();

  // predictor: S* and store ω_n
  dm_predict<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), spins_initial_.mutable_device_data(),
      omega1_.mutable_device_data(), globals::h.device_data(), gyro, field_scale, alpha,
      globals::num_spins, dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  record_spin_barrier_event();

  time_ = step_start_time + dt;
  compute_fields();

  // correct: rotate original S_n using averaged ω, output into globals::s
  dm_correct<<<grid_size, block_size, 0, stream>>>(
      spins_initial_.device_data(), omega1_.device_data(), globals::s.mutable_device_data(),
      globals::h.device_data(), gyro, field_scale, alpha, globals::num_spins, dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  // Thermal half-step
  update_thermostat_on_stream(stream);

  cuda_llg_noise_step_rodrigues_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), thermostat_->device_data(), gyro, alpha, globals::num_spins,
      half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);
  record_spin_barrier_event();

  advance_time();
}
