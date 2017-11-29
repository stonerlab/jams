// Copyright 2014 Joseph Barker. All rights reserved.

#include "cuda_llg_heun.h"

#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <jams/interface/config.h>
#include <jams/helpers/defaults.h>

#include "jams/helpers/consts.h"
#include "jams/helpers/exception.h"
#include "jams/cuda/cuda_sparsematrix.h"
#include "jams/core/globals.h"
#include "jams/core/thermostat.h"
#include "jams/core/output.h"

#include "cuda_llg_heun_kernel.h"

#include "jblib/containers/array.h"

using namespace std;

void CUDAHeunLLGSolver::initialize(const libconfig::Setting& settings)
{
  using namespace globals;

  CudaSolver::initialize(settings);

  time_step_ = jams::config_required<double>(settings, "t_step");
  double dt = time_step_ * kGyromagneticRatio;

  auto t_max = jams::config_required<double>(settings, "t_max");
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0);

  max_steps_ = static_cast<int>(t_max / time_step_);
  min_steps_ = static_cast<int>(t_min / time_step_);

  cout << "timestep " << dt << "\n";
  cout << "t_max " << t_max << " steps (" <<  max_steps_ << ")\n";
  cout << "t_min " << t_min << " steps (" << min_steps_ << ")\n";

  cout << "  creating stream\n";
  if(cudaStreamCreate(&dev_stream_) != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  cout << "  copy time_step to symbol\n";
  if(cudaMemcpyToSymbol(dev_dt, &dt, sizeof(double)) != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  cout << "  copy num_spins to symbol\n";
  if(cudaMemcpyToSymbol(dev_num_spins, &globals::num_spins, sizeof(unsigned int)) != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  std::string thermostat_name = jams::config_optional<string>(config->lookup("sim"), "thermostat", jams::default_solver_gpu_thermostat);
  thermostat_ = Thermostat::create(thermostat_name);

  cout << "  thermostat " << thermostat_name.c_str() << "\n";

  nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;

  cout << "done\n";
}

void CUDAHeunLLGSolver::run()
{
  using namespace globals;

  dim3 block_size;
  block_size.x = 85;
  block_size.y = 3;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;
  grid_size.y = (3 + block_size.y - 1) / block_size.y;

    cudaMemcpyAsync(dev_s_old_.data(),           // void *               dst
               dev_s_.data(),               // const void *         src
               num_spins3*sizeof(double),   // size_t               count
               cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
               dev_stream_);                   // device stream

  if (debug_is_enabled()) {
    if (cudaPeekAtLastError() != cudaSuccess) {
     throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
  }



    thermostat_->set_temperature(physics_module_->temperature());
    thermostat_->update();

    compute_fields();

    cuda_heun_llg_kernelA<<<grid_size, block_size>>>
        (dev_s_.data(), dev_ds_dt_.data(), dev_s_old_.data(),
          dev_h_.data(), thermostat_->noise(),
          dev_gyro_.data(), dev_alpha_.data());

  if (debug_is_enabled()) {
    if (cudaPeekAtLastError() != cudaSuccess) {
      throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
  }

    compute_fields();

    cuda_heun_llg_kernelB<<<grid_size, block_size>>>
      (dev_s_.data(), dev_ds_dt_.data(), dev_s_old_.data(),
        dev_h_.data(), thermostat_->noise(),
        dev_gyro_.data(), dev_alpha_.data());

  if (debug_is_enabled()) {
    if (cudaPeekAtLastError() != cudaSuccess) {
      throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
  }


    iteration_++;
}


CUDAHeunLLGSolver::~CUDAHeunLLGSolver()
{
  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }
}

