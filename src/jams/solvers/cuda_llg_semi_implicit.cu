//
// Created by Joseph Barker on 05/01/2026.
//

#include "jams/solvers/cuda_llg_semi_implicit.h"

#include <iostream>
#include <string>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/helpers/defaults.h"
#include "jams/interface/config.h"
#include "jams/solvers/cuda_solver_functions.cuh"

namespace {

template <typename DtGyroMuParam, typename AlphaParam>
__global__ __maxnreg__(32) void semi_implicit_predict(
    double* __restrict__ spins,  // in: S_n, out: (S_n + S'_n+1) / 2
    double* __restrict__ initial_spins, const jams::Real* __restrict__ fields,
    const DtGyroMuParam dt_gyro_mu, const AlphaParam alpha, const unsigned num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned int offset = 3u * spin_index;

  const jams::Real h[3] = {fields[offset], fields[offset + 1], fields[offset + 2]};
  const double s[3] = {spins[offset], spins[offset + 1], spins[offset + 2]};
  double omega[3];
  omega_llg(s, h, dt_gyro_mu.get(spin_index), alpha.get(spin_index), omega);
  project_to_tangent(omega, s);

  double s_pred[3];
  cayley_rotate(omega, s, s_pred);
  for (int n = 0; n < 3; ++n) {
    initial_spins[offset + n] = s[n];
    spins[offset + n] = 0.5 * (s_pred[n] + s[n]);
  }
}

template <typename DtGyroMuParam, typename AlphaParam>
__global__ void semi_implicit_correct(
    double* __restrict__ spins,                // in: (S_n + S'_{n+1}) / 2 out: S_{n+1}
    const double* __restrict__ initial_spins,  // S_n
    const jams::Real* __restrict__ fields,     // field at the same time as s_step
    const DtGyroMuParam dt_gyro_mu, const AlphaParam alpha, const unsigned num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned int offset = 3u * spin_index;

  const jams::Real h[3] = {fields[offset], fields[offset + 1], fields[offset + 2]};
  const double s[3] = {spins[offset], spins[offset + 1], spins[offset + 2]};
  double omega[3];
  omega_llg(s, h, dt_gyro_mu.get(spin_index), alpha.get(spin_index), omega);
  project_to_tangent(omega, s);

  const double s_initial[3] = {
      initial_spins[offset], initial_spins[offset + 1], initial_spins[offset + 2]};
  double s_out[3];
  cayley_rotate(omega, s_initial, s_out);
  for (int n = 0; n < 3; ++n) {
    spins[offset + n] = s_out[n];
  }
}

}  // namespace

void CUDALLGSemiImplictSolver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);

  const auto thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name, 0.5 * time_step()));

  std::cout << "  thermostat " << thermostat_name << "\n";

  initialize_gyro_eff(settings, gyro_eff_);
  const auto gyro_eff_choice = cuda_spin_parameter_choice(gyro_eff_);
  gyro_eff_is_uniform_ = gyro_eff_choice.is_uniform;
  gyro_eff_uniform_value_ = gyro_eff_choice.uniform_value;
  const auto alpha_choice = cuda_spin_parameter_choice(globals::alpha);
  alpha_is_uniform_ = alpha_choice.is_uniform;
  alpha_uniform_value_ = alpha_choice.uniform_value;

  dt_gyro_mu_.resize(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    dt_gyro_mu_(i) = step_size_ * gyro_eff_(i) * globals::inv_mus(i);
  }
  const auto dt_gyro_mu_choice = cuda_spin_parameter_choice(dt_gyro_mu_);
  dt_gyro_mu_is_uniform_ = dt_gyro_mu_choice.is_uniform;
  dt_gyro_mu_uniform_value_ = dt_gyro_mu_choice.uniform_value;

  spins_initial_.resize(globals::num_spins, 3);
}

void CUDALLGSemiImplictSolver::run() {
  dispatch_cuda_spin_parameter(
      {gyro_eff_is_uniform_, gyro_eff_uniform_value_}, gyro_eff_, [this](const auto gyro) {
        dispatch_cuda_spin_parameters({dt_gyro_mu_is_uniform_, dt_gyro_mu_uniform_value_},
                                      {alpha_is_uniform_, alpha_uniform_value_}, dt_gyro_mu_,
                                      globals::alpha,
                                      [this, gyro](const auto dt_gyro_mu, const auto alpha) {
                                        run_with_parameters(gyro, alpha, dt_gyro_mu);
                                      });
      });
}

template <typename GyroParam, typename AlphaParam, typename DtGyroMuParam>
void CUDALLGSemiImplictSolver::run_with_parameters(const GyroParam gyro, const AlphaParam alpha,
                                                   const DtGyroMuParam dt_gyro_mu) {
  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;
  const auto stream = jams::instance().cuda_master_stream().get();

  const dim3 block_size = {64, 1, 1};
  const auto grid_size =
      cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  update_thermostat_on_stream(stream);

  cuda_llg_noise_step_cayley_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), thermostat_->device_data(), gyro, alpha, globals::num_spins,
      half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);
  record_spin_barrier_event();

  compute_fields();

  semi_implicit_predict<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), spins_initial_.mutable_device_data(),
      globals::h.device_data(), dt_gyro_mu, alpha, globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  time_ = step_start_time + half_dt;
  compute_fields();

  semi_implicit_correct<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), spins_initial_.device_data(), globals::h.device_data(),
      dt_gyro_mu, alpha, globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat_on_stream(stream);

  cuda_llg_noise_step_cayley_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), thermostat_->device_data(), gyro, alpha, globals::num_spins,
      half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);
  record_spin_barrier_event();

  advance_time();
}
