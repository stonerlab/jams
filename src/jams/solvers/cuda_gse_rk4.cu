// cuda_gse_rk4.cu                                                     -*-C++-*-

#include <jams/solvers/cuda_gse_rk4.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include <jams/common.h>
#include <jams/interface/config.h>
#include <jams/helpers/defaults.h>

#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/thermostat.h"
#include "jams/core/physics.h"
#include "jams/helpers/error.h"
#include "jams/cuda/cuda_common.h"

#include "cuda_gse_rk4_kernel.cuh"

void CUDAGSERK4Solver::initialize(const libconfig::Setting& settings)
{
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

void CUDAGSERK4Solver::run()
{
  double t0 = time_;

  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cudaMemcpyAsync(s_old_.device_data(),           // void *               dst
                  globals::s.device_data(),               // const void *         src
                  globals::num_spins3*sizeof(double),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  dev_stream_.get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();

  compute_fields();

  // k1
  cuda_gse_rk4_kernel<<<grid_size, block_size>>>
      (globals::s.device_data(), k1_.device_data(),
       globals::h.device_data(), thermostat_->device_data(),
       globals::gyro.device_data(), globals::mus.device_data(),
       globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  double mid_time_step = 0.5 * step_size_;
  time_ = t0 + mid_time_step;

  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, globals::s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k1_.device_data(), 1, globals::s.device_data(), 1));

  compute_fields();

  // k2
  cuda_gse_rk4_kernel<<<grid_size, block_size>>>
      (globals::s.device_data(), k2_.device_data(),
       globals::h.device_data(), thermostat_->device_data(),
       globals::gyro.device_data(), globals::mus.device_data(),
       globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

      mid_time_step = 0.5 * step_size_;
  time_ = t0 + mid_time_step;

  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, globals::s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k2_.device_data(), 1, globals::s.device_data(), 1));

  compute_fields();

  // k3
  cuda_gse_rk4_kernel<<<grid_size, block_size>>>
      (globals::s.device_data(), k3_.device_data(),
       globals::h.device_data(), thermostat_->device_data(),
       globals::gyro.device_data(), globals::mus.device_data(),
       globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

      mid_time_step = step_size_;
  time_ = t0 + mid_time_step;

  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, globals::s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k3_.device_data(), 1, globals::s.device_data(), 1));

  compute_fields();

  // k4
  cuda_gse_rk4_kernel<<<grid_size, block_size>>>
      (globals::s.device_data(), k4_.device_data(),
       globals::h.device_data(), thermostat_->device_data(),
       globals::gyro.device_data(), globals::mus.device_data(),
       globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

      cuda_gse_rk4_combination_kernel<<<grid_size, block_size>>>
      (globals::s.device_data(), s_old_.device_data(),
       k1_.device_data(), k2_.device_data(), k3_.device_data(), k4_.device_data(),
       step_size_, globals::num_spins);

  iteration_++;
  time_ = iteration_ * step_size_;
}

