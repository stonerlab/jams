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
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;


  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  std::cout << "\ntimestep (ps) " << step_size_ << "\n";
  std::cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  std::cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  std::cout << "timestep " << step_size_ << "\n";
  std::cout << "t_max " << t_max << " steps (" <<  max_steps_ << ")\n";
  std::cout << "t_min " << t_min << " steps (" << min_steps_ << ")\n";

  std::string thermostat_name = jams::config_optional<std::string>(globals::config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name, this->time_step()));

  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  std::cout << "done\n";

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


void CudaRK4BaseSolver::run()
{
  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins3), 1, 1});

  double t0 = time_;

  cudaMemcpyAsync(s_old_.device_data(),           // void *               dst
                  globals::s.device_data(),               // const void *         src
                  globals::num_spins3*sizeof(double),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  jams::instance().cuda_master_stream().get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  pre_step(globals::s);

  update_thermostat();
  thermostat_->record_done();
  thermostat_->wait_on(jams::instance().cuda_master_stream().get());


  // k1
  record_spin_barrier_event();
  function_kernel(globals::s, k1_);

  double mid_time_step = 0.5 * step_size_;
  time_ = t0 + mid_time_step;

  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0 ,jams::instance().cuda_master_stream().get()>>>(globals::num_spins3, mid_time_step, s_old_.device_data(), k1_.device_data(), globals::s.device_data());
  // CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, globals::s.device_data(), 1));
  // CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k1_.device_data(), 1, globals::s.device_data(), 1));

  record_spin_barrier_event();
  function_kernel(globals::s, k2_);

  mid_time_step = 0.5 * step_size_;
  time_ = t0 + mid_time_step;

  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0 ,jams::instance().cuda_master_stream().get()>>>(globals::num_spins3, mid_time_step, s_old_.device_data(), k2_.device_data(), globals::s.device_data());
  // CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, globals::s.device_data(), 1));
  // CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k2_.device_data(), 1, globals::s.device_data(), 1));

  record_spin_barrier_event();
  function_kernel(globals::s, k3_);

  mid_time_step = step_size_;
  time_ = t0 + mid_time_step;

  cuda_rk4_mid_step_kernel<<<grid_size, block_size, 0 ,jams::instance().cuda_master_stream().get()>>>(globals::num_spins3, mid_time_step, s_old_.device_data(), k3_.device_data(), globals::s.device_data());
  // CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, globals::s.device_data(), 1));
  // CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k3_.device_data(), 1, globals::s.device_data(), 1));

  record_spin_barrier_event();
  function_kernel(globals::s, k4_);


  // NOTE: this does NOT normalise the spins. This must be done in the post_step
  // function
  record_spin_barrier_event();
  cuda_rk4_combination_kernel<<<grid_size, block_size, 0 ,jams::instance().cuda_master_stream().get()>>>
      (globals::s.device_data(), s_old_.device_data(),
       k1_.device_data(), k2_.device_data(), k3_.device_data(), k4_.device_data(),
       step_size_, globals::num_spins3);

  record_spin_barrier_event();
  post_step(globals::s);

  record_spin_barrier_event();

  iteration_++;
  time_ = iteration_ * step_size_;
}