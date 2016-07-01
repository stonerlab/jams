// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/cuda_heunllg.h"

#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>

#include "core/consts.h"
#include "core/exception.h"
#include "core/cuda_sparsematrix.h"
#include "core/globals.h"
#include "core/thermostat.h"

#include "solvers/cuda_heunllg_kernel.h"

#include "jblib/containers/array.h"

void CUDAHeunLLGSolver::initialize(int argc, char **argv, double idt)
{
  using namespace globals;

  CudaSolver::initialize(argc, argv, idt);

  ::output.write("\ninitializing CUDA Heun LLG solver\n");

  if (::config.exists("sim.thermostat")) {
    thermostat_ = Thermostat::create(::config.lookup("sim.thermostat"));
  } else {
    ::output.write("  DEFAULT thermostat\n");
    thermostat_ = Thermostat::create("CUDA_LANGEVIN_WHITE");
  }

  nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;

  ::output.write("\n");

  if(cudaStreamCreate(&dev_stream_) != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  if(cudaMemcpyToSymbol(dev_dt, &time_step_, sizeof(double)) != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  if(cudaMemcpyToSymbol(dev_num_spins, &globals::num_spins, sizeof(unsigned int)) != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }
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

#ifdef DEBUG
  if (cudaPeekAtLastError() != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }
#endif



    thermostat_->set_temperature(physics_module_->temperature());
    thermostat_->update();

    compute_fields();

    cuda_heun_llg_kernelA<<<grid_size, block_size>>>
        (dev_s_.data(), dev_ds_dt_.data(), dev_s_old_.data(),
          dev_h_.data(), thermostat_->noise(),
          dev_gyro_.data(), dev_alpha_.data());

#ifdef DEBUG
    if (cudaPeekAtLastError() != cudaSuccess) {
      throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
#endif

    compute_fields();

    cuda_heun_llg_kernelB<<<grid_size, block_size>>>
      (dev_s_.data(), dev_ds_dt_.data(), dev_s_old_.data(),
        dev_h_.data(), thermostat_->noise(),
        dev_gyro_.data(), dev_alpha_.data());

#ifdef DEBUG
    if (cudaPeekAtLastError() != cudaSuccess) {
      throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
#endif


    iteration_++;
}


CUDAHeunLLGSolver::~CUDAHeunLLGSolver()
{
}

