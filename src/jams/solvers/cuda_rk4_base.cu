#include "jams/solvers/cuda_rk4_base.h"

#include <iostream>
#include <string>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/cuda/cuda_common.h"
#include "jams/helpers/defaults.h"
#include "jams/interface/config.h"
#include "jams/solvers/cuda_rk4_base_kernel.cuh"

void CudaRK4BaseSolver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);

  const auto thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name, time_step()));

  std::cout << "  thermostat " << thermostat_name << "\n";

  initialize_gyro_eff(settings, gyro_eff_);

  spins_initial_.resize(globals::num_spins, 3);

  k1_.resize(globals::num_spins, 3);
  k2_.resize(globals::num_spins, 3);
  k3_.resize(globals::num_spins, 3);
  k4_.resize(globals::num_spins, 3);
}

void CudaRK4BaseSolver::run() {
  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;
  const auto stream = jams::instance().cuda_master_stream().get();

  const dim3 block_size = {64, 1, 1};
  const auto grid_size =
      cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins3), 1, 1});

  CHECK_CUDA_STATUS(cudaMemcpyAsync(spins_initial_.mutable_device_data(), globals::s.device_data(),
                                    globals::s.bytes(), cudaMemcpyDeviceToDevice, stream));
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat_on_stream(stream);

  evaluate_stage(k1_, stream);

  time_ = step_start_time + half_dt;
  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::num_spins3, half_dt, spins_initial_.device_data(), k1_.device_data(),
      globals::s.mutable_device_data());
  evaluate_stage(k2_, stream);

  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::num_spins3, half_dt, spins_initial_.device_data(), k2_.device_data(),
      globals::s.mutable_device_data());
  evaluate_stage(k3_, stream);

  time_ = step_start_time + dt;
  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::num_spins3, dt, spins_initial_.device_data(), k3_.device_data(),
      globals::s.mutable_device_data());
  evaluate_stage(k4_, stream);
  thermostat_->record_consumed(stream);

  cuda_rk4_combination_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), spins_initial_.device_data(), k1_.device_data(),
      k2_.device_data(), k3_.device_data(), k4_.device_data(), dt, globals::num_spins3);

  post_step(globals::s, stream);
  record_spin_barrier_event();

  advance_time();
}

void CudaRK4BaseSolver::evaluate_stage(jams::MultiArray<double, 2>& k, cudaStream_t stream) {
  pre_stage(globals::s, stream);
  record_spin_barrier_event();
  compute_fields();
  function_kernel(globals::s, k, stream);
}
