// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/cuda_heunllg.h"

#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>

#include "core/consts.h"
#include "core/cuda_sparsematrix.h"
#include "core/globals.h"

#include "solvers/cuda_heunllg_kernel.h"

#include "jblib/containers/array.h"

void CUDAHeunLLGSolver::initialize(int argc, char **argv, double idt)
{
  using namespace globals;

  CudaSolver::initialize(argc, argv, idt);

  ::output.write("\ninitializing CUDA Heun LLG solver");

  thermostat_ = Thermostat::create("CUDA_LANGEVIN_WHITE");
  nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;

  ::output.write("\n");
}

void CUDAHeunLLGSolver::run()
{
  using namespace globals;

    thermostat_->set_temperature(physics_module_->temperature());
    thermostat_->update();

    compute_fields();

    cuda_heun_llg_kernelA<<<nblocks, BLOCKSIZE>>>
        (dev_s_.data(), dev_s_new_.data(), dev_h_.data(), thermostat_->noise(), dev_mat_.data(), physics_module_->applied_field(0), physics_module_->applied_field(1), physics_module_->applied_field(2), num_spins, time_step_);

    compute_fields();

    cuda_heun_llg_kernelB<<<nblocks, BLOCKSIZE>>>
        (dev_s_.data(), dev_s_new_.data(), dev_h_.data(), thermostat_->noise(), dev_mat_.data(), physics_module_->applied_field(0), physics_module_->applied_field(1), physics_module_->applied_field(2), num_spins, time_step_);

    iteration_++;
}

void CUDAHeunLLGSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s){

}

CUDAHeunLLGSolver::~CUDAHeunLLGSolver()
{
}

