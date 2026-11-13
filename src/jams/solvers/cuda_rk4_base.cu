// cuda_rk4_base.cpp                                                          -*-C++-*-
#include <jams/solvers/cuda_rk4_base.h>
#include <jams/solvers/cuda_rk4_base_kernel.cuh>

#include <jams/interface/config.h>
#include <jams/core/globals.h>
#include <jams/helpers/defaults.h>
#include <jams/common.h>
#include <jams/cuda/cuda_common.h>

void CudaRK4BaseSolver::initialize(const libconfig::Setting &settings) {
  // convert input in seconds to picoseconds for internal units
  initialize_time(settings);

  std::string thermostat_name = jams::config_optional<std::string>(globals::config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name, this->time_step()));

  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  std::cout << "done\n";

  initialize_gyro_eff(settings, gyro_eff_);

  s_old_.resize(globals::num_spins, 3);
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_old_(i, j) = globals::s(i, j);
    }
  }

  k1_.resize(globals::num_spins, 3);
  k2_.resize(globals::num_spins, 3);
  k3_.resize(globals::num_spins, 3);
  k4_.resize(globals::num_spins, 3);
}


void CudaRK4BaseSolver::evaluate_stage(jams::MultiArray<double, 2>& k, cudaStream_t stream) {
  pre_stage(globals::s, stream);
  record_spin_barrier_event();
  compute_fields();
  function_kernel(globals::s, k, stream);
}


void CudaRK4BaseSolver::run()
{
  const auto stream = jams::instance().cuda_master_stream().get();
  const dim3 block_size = {64, 1, 1};
  const auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins3), 1, 1});

  const double t0 = time_;
  const double half_step = 0.5 * step_size_;

  CHECK_CUDA_STATUS(cudaMemcpyAsync(s_old_.mutable_device_data(),
      globals::s.device_data(), globals::s.bytes(), cudaMemcpyDeviceToDevice, stream));
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();
  thermostat_->record_done();
  thermostat_->wait_on(stream);

  evaluate_stage(k1_, stream);

  time_ = t0 + half_step;
  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0, stream>>>(globals::num_spins3,
      half_step, s_old_.device_data(), k1_.device_data(), globals::s.mutable_device_data());
  evaluate_stage(k2_, stream);

  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0, stream>>>(globals::num_spins3,
      half_step, s_old_.device_data(), k2_.device_data(), globals::s.mutable_device_data());
  evaluate_stage(k3_, stream);

  time_ = t0 + step_size_;
  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0, stream>>>(globals::num_spins3,
      step_size_, s_old_.device_data(), k3_.device_data(), globals::s.mutable_device_data());
  evaluate_stage(k4_, stream);
  thermostat_->record_consumed(stream);

  cuda_rk4_combination_kernel<<<grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), s_old_.device_data(),
      k1_.device_data(), k2_.device_data(), k3_.device_data(), k4_.device_data(),
      step_size_, globals::num_spins3);

  post_step(globals::s, stream);
  record_spin_barrier_event();

  advance_time();
}
