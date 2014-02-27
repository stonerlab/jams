// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/cuda_heunllg.h"

#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>

#include "core/consts.h"
#include "core/cuda_sparse.h"
#include "core/cuda_sparse_types.h"
#include "core/globals.h"

#include "solvers/cuda_heunllg_kernel.h"

#include "jblib/containers/array.h"

void CUDAHeunLLGSolver::sync_device_data()
{
  using namespace globals;
  s_dev.copy_to_host_array(s);
}

void CUDAHeunLLGSolver::initialize(int argc, char **argv, double idt)
{
  using namespace globals;

  Solver::initialize(argc, argv, idt);

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

  output.write("  * Converting sparse matrices\n");
  J1ij_t.convertMAP2DIA();

  output.write("    - J1ij tensor matrix memory (DIA): %f MB\n", J1ij_t.calculateMemory());


  J1ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
  allocate_transfer_dia(J1ij_t, J1ij_t_dev);

  output.write("  * Allocating device memory...\n");

  // temporary host arrays for device data
  jblib::Array<float, 2> zero(num_spins, 3, 0.0);
  jblib::Array<float, 2> sf(num_spins, 3);
  jblib::Array<float, 2> mat(num_spins, 4);

  sigma.resize(num_spins);
  for(int i = 0; i!=num_spins; ++i) {
    sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)) / (time_step_*mus(i)*mu_bohr_si) );
  }

  for(int i = 0; i!=num_spins; ++i) {
    for(int j = 0; j!=3; ++j) {
      sf(i, j) = static_cast<float>(s(i, j));
    }
  }

  for(int i = 0; i!=num_spins; ++i){
    mat(i, 0) = mus(i);   mat(i, 1) = gyro(i);
    mat(i, 2) = alpha(i); mat(i, 3) = sigma(i);
  }

  // allocate and initialize
  s_dev        = jblib::CudaArray<double, 1>(s);
  s_new_dev    = jblib::CudaArray<double, 1>(s);
  sf_dev       = jblib::CudaArray<float, 1>(sf);
  h_dev        = jblib::CudaArray<float, 1>(zero);
  e_dev        = jblib::CudaArray<float, 1>(zero);
  mat_dev      = jblib::CudaArray<float, 1>(mat);

  jblib::Array<float, 1> w(num_spins3+(num_spins3%2), 0.0);
  w_dev = jblib::CudaArray<float, 1>(w);

  eng.resize(num_spins, 3);

  nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;
}

void CUDAHeunLLGSolver::run()
{
  using namespace globals;

    float stmp = sqrt(physics_module_->temperature());

    if(physics_module_->temperature() > 0.0) {
        CURAND_CALL(curandGenerateNormal(gen, w_dev.data(), (num_spins3+(num_spins3%2)), 0.0f, stmp));
    }

    compute_fields();

    cuda_heun_llg_kernelA<<<nblocks, BLOCKSIZE>>>
        (s_dev.data(), sf_dev.data(), s_new_dev.data(), h_dev.data(), w_dev.data(), mat_dev.data(), physics_module_->applied_field(0), physics_module_->applied_field(1), physics_module_->applied_field(2), num_spins, time_step_);

    compute_fields();

    cuda_heun_llg_kernelB<<<nblocks, BLOCKSIZE>>>
        (s_dev.data(), sf_dev.data(), s_new_dev.data(), h_dev.data(), w_dev.data(), mat_dev.data(), physics_module_->applied_field(0), physics_module_->applied_field(1), physics_module_->applied_field(2), num_spins, time_step_);

    iteration_++;
    sync_device_data();
}

void CUDAHeunLLGSolver::compute_fields() {
  using namespace globals;

// bilinear interactions
  if(J1ij_t.nonZero() > 0){

    spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>
    (num_spins3, num_spins3, J1ij_t.diags(), J1ij_t_dev.pitch, 1.0, 0.0,
     J1ij_t_dev.row, J1ij_t_dev.val, sf_dev.data(), h_dev.data());
  }

}

void CUDAHeunLLGSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s){
  using namespace globals;

  e1_s = 0.0; e1_t = 0.0; e2_s = 0.0; e2_t = 0.0;

}

CUDAHeunLLGSolver::~CUDAHeunLLGSolver()
{
  curandDestroyGenerator(gen);

  cusparseStatus_t status;

  status = cusparseDestroyMatDescr(descra);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE matrix destruction failed");
  }
  CUDA_CALL(cudaThreadSynchronize());

  status = cusparseDestroy(handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Library destruction failed");
  }
  CUDA_CALL(cudaThreadSynchronize());

  //-------------------------------------------------------------------
  //  Free device memory
  //-------------------------------------------------------------------

  free_dia(J1ij_t_dev);
}

