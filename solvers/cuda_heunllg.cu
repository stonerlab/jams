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

  output.write("  * CUDA Heun LLG solver (GPU)\n");

  output.write("  * Initialising CURAND...\n");

  const unsigned long long gpuseed = rng.uniform()*18446744073709551615ULL;

  if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to create CURAND generator");
  }

  if (curandSetPseudoRandomGeneratorSeed(gen, gpuseed) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to set CURAND seed");
  }

  if (curandGenerateSeeds(gen) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to generate CURAND seeds");
  }



  jblib::Array<CudaFastFloat, 1> w(num_spins3+(num_spins3%2), 0.0);
  dev_w_ = jblib::CudaArray<CudaFastFloat, 1>(w);

  nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;
}

void CUDAHeunLLGSolver::run()
{
  using namespace globals;

    CudaFastFloat stmp = sqrt(physics_module_->temperature());

    if(physics_module_->temperature() > 0.0) {
        CURAND_CALL(curandGenerateNormal(gen, dev_w_.data(), (num_spins3+(num_spins3%2)), 0.0f, stmp));
    }

    compute_fields();

    cuda_heun_llg_kernelA<<<nblocks, BLOCKSIZE>>>
        (dev_s_.data(), dev_s_float_.data(), dev_s_new_.data(), dev_h_.data(), dev_w_.data(), dev_mat_.data(), physics_module_->applied_field(0), physics_module_->applied_field(1), physics_module_->applied_field(2), num_spins, time_step_);

    compute_fields();

    cuda_heun_llg_kernelB<<<nblocks, BLOCKSIZE>>>
        (dev_s_.data(), dev_s_float_.data(), dev_s_new_.data(), dev_h_.data(), dev_w_.data(), dev_mat_.data(), physics_module_->applied_field(0), physics_module_->applied_field(1), physics_module_->applied_field(2), num_spins, time_step_);

    iteration_++;
}

void CUDAHeunLLGSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s){

}

CUDAHeunLLGSolver::~CUDAHeunLLGSolver()
{
  curandDestroyGenerator(gen);

  //-------------------------------------------------------------------
  //  Free device memory
  //-------------------------------------------------------------------

}

