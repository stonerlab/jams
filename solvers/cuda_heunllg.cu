// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/cuda_heunllg.h"

#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>

#include "core/consts.h"
#include "core/cuda_fields.h"
#include "core/cuda_sparse.h"
#include "core/cuda_sparse_types.h"
#include "core/globals.h"

#include "solvers/cuda_heunllg_kernel.h"

#include "jblib/containers/array.h"


void CUDAHeunLLGSolver::sync_device_data()
{
  using namespace globals;
  s_dev.copyToHostArray(s);
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
  J1ij_s.convertMAP2DIA();
  J1ij_t.convertMAP2DIA();
  J2ij_s.convertMAP2DIA();
  J2ij_t.convertMAP2DIA();
  output.write("    - J1ij scalar matrix memory (DIA): %f MB\n", J1ij_s.calculateMemory());
  output.write("    - J1ij tensor matrix memory (DIA): %f MB\n", J1ij_t.calculateMemory());
  output.write("    - J2ij scalar matrix memory (DIA): %f MB\n", J2ij_s.calculateMemory());
  output.write("    - J2ij tensor matrix memory (DIA): %f MB\n", J2ij_t.calculateMemory());
  output.write("    - J4ijkl scalar matrix memory (CSR): %f MB\n", J4ijkl_s.calculateMemoryUsage());

  output.write("  * Allocating device memory...\n");

  // temporary host arrays for device data
  jblib::Array<float, 2> zero(nspins, 3, 0.0);
  jblib::Array<float, 2> sf(nspins, 3);
  jblib::Array<float, 2> mat(nspins, 4);
  jblib::Array<float, 1> r_maxf(3);
  jblib::Array<bool, 1> pbc(3);

  sigma.resize(nspins);
  for(int i = 0; i!=nspins; ++i) {
    sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)) / (dt*mus(i)*mu_bohr_si) );
  }

  for(int i = 0; i!=nspins; ++i) {
    for(int j = 0; j!=3; ++j) {
      sf(i, j) = static_cast<float>(s(i, j));
    }
  }

  for(int i = 0; i!=nspins; ++i){
    mat(i, 0) = mus(i);   mat(i, 1) = gyro(i);
    mat(i, 2) = alpha(i); mat(i, 3) = sigma(i);
  }

  lattice.getMaxDimensions(r_maxf(0), r_maxf(1), r_maxf(2));
  lattice.getBoundaries(pbc(0), pbc(1), pbc(2));

  // allocate and initialize
  s_dev        = jblib::CudaArray<double, 1>(s);
  s_new_dev    = jblib::CudaArray<double, 1>(s);
  sf_dev       = jblib::CudaArray<float, 1>(sf);
  r_dev        = jblib::CudaArray<float, 1>(atom_pos);
  h_dev        = jblib::CudaArray<float, 1>(zero);
  e_dev        = jblib::CudaArray<float, 1>(zero);
  h_dipole_dev = jblib::CudaArray<float, 1>(zero);
  mat_dev      = jblib::CudaArray<float, 1>(mat);
  r_max_dev    = jblib::CudaArray<float, 1>(r_maxf);
  pbc_dev      = jblib::CudaArray<bool, 1>(pbc);

  w_dev.resize(nspins3+(nspins3%2));
  eng.resize(nspins, 3);

  allocate_transfer_dia(J1ij_s, J1ij_s_dev);
  allocate_transfer_dia(J1ij_t, J1ij_t_dev);
  allocate_transfer_dia(J2ij_s, J2ij_s_dev);
  allocate_transfer_dia(J2ij_t, J2ij_t_dev);
  allocate_transfer_csr_4d(J4ijkl_s, J4ijkl_s_dev);

  nblocks = (nspins+BLOCKSIZE-1)/BLOCKSIZE;

  J1ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (nspins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
  J1ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (nspins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

  J2ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (nspins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
  J2ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (nspins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

  J4ijkl_s_dev.blocks = std::min<int>(CSR_4D_BLOCK_SIZE, (nspins+CSR_4D_BLOCK_SIZE-1)/CSR_4D_BLOCK_SIZE);

  initialized = true;
}

void CUDAHeunLLGSolver::run()
{
  using namespace globals;

    float stmp = sqrt(temperature);

    if(temperature > 0.0) {
        CURAND_CALL(curandGenerateNormal(gen, w_dev.data(), (nspins3+(nspins3%2)), 0.0f, stmp));
    }

    cuda_device_compute_fields(J1ij_s_dev, J1ij_t_dev, J2ij_s_dev, J2ij_t_dev, J4ijkl_s_dev, sf_dev.data(), r_dev.data(), r_max_dev.data(), mat_dev.data(), pbc_dev.data(), h_dev.data(), h_dipole_dev.data(), true);

    cuda_heun_llg_kernelA<<<nblocks, BLOCKSIZE>>>
        (s_dev.data(), sf_dev.data(), s_new_dev.data(), h_dev.data(), w_dev.data(), mat_dev.data(), h_app[0], h_app[1], h_app[2], nspins, dt);

    cuda_device_compute_fields(J1ij_s_dev, J1ij_t_dev, J2ij_s_dev, J2ij_t_dev, J4ijkl_s_dev, sf_dev.data(), r_dev.data(), r_max_dev.data(), mat_dev.data(), pbc_dev.data(), h_dev.data(), h_dipole_dev.data(), false);

    cuda_heun_llg_kernelB<<<nblocks, BLOCKSIZE>>>
        (s_dev.data(), sf_dev.data(), s_new_dev.data(), h_dev.data(), w_dev.data(), mat_dev.data(), h_app[0], h_app[1], h_app[2], nspins, dt);

    iteration++;
}

void CUDAHeunLLGSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s){
  using namespace globals;
  const float beta=0.0;

  e1_s = 0.0; e1_t = 0.0; e2_s = 0.0; e2_t = 0.0;
  /*
  //size_t offset = size_t(-1);
  //CUDA_CALL(cudaBindTexture(&offset, tex_x_float, sf_dev));

  // bilinear scalar
  if(J1ij_s.nonZero() > 0){
    bilinear_scalar_interaction_dia_kernel<<< J1ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(nspins, nspins,
      J1ij_s.diags(), J1ij_s_dev.pitch, 1.0, beta, J1ij_s_dev.row, J1ij_s_dev.val, sf_dev.data(), e_dev.data());
    CUDA_CALL(cudaMemcpy(eng.data(), e_dev, (size_t)(nspins3*sizeof(float)), cudaMemcpyDeviceToHost));
    for(int i = 0; i<nspins; ++i){
      e1_s = e1_s + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }
    e1_s = e1_s/nspins;
  }


  // bilinear tensor
  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(nspins3, nspins3,
      J1ij_t.diags(), J1ij_t_dev.pitch, 1.0, beta, J1ij_t_dev.row, J1ij_t_dev.val, sf_dev, e_dev);
    CUDA_CALL(cudaMemcpy(eng.data(), e_dev, (size_t)(nspins3*sizeof(float)), cudaMemcpyDeviceToHost));
    for(int i = 0; i<nspins; ++i){
      e1_t = e1_t + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }
    e1_t = e1_t/nspins;
  }


  // biquadratic scalar
  if(J2ij_s.nonZero() > 0){
    biquadratic_scalar_dia_kernel<<< J2ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(nspins, nspins,
      J2ij_s.diags(), J2ij_s_dev.pitch, 1.0, beta, J2ij_s_dev.row, J2ij_s_dev.val, sf_dev, e_dev);
    CUDA_CALL(cudaMemcpy(eng.data(), e_dev, (size_t)(nspins3*sizeof(float)), cudaMemcpyDeviceToHost));
    for(int i = 0; i<nspins; ++i){
      e2_s = e2_s + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }

    e2_s = e2_s/nspins;
  }

  // biquadratic tensor
  if(J2ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J2ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(nspins3, nspins3,
      J2ij_t.diags(), J2ij_t_dev.pitch, 1.0, beta, J2ij_t_dev.row, J2ij_t_dev.val, sf_dev, e_dev);
    CUDA_CALL(cudaMemcpy(eng.data(), e_dev, (size_t)(nspins3*sizeof(float)), cudaMemcpyDeviceToHost));

    for(int i = 0; i<nspins; ++i){
      e2_t = e2_t + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }

    e2_t = e2_t/nspins;
  }

  if(J4ijkl_s.nonZeros() > 0){
    fourspin_scalar_interaction_csr_kernel<<< J4ijkl_s_dev.blocks, CSR_4D_BLOCK_SIZE>>>(nspins, nspins, 1.0, beta,
        J4ijkl_s_dev.pointers, J4ijkl_s_dev.coords, J4ijkl_s_dev.val, sf_dev, e_dev);
    CUDA_CALL(cudaMemcpy(eng.data(), e_dev, (size_t)(nspins3*sizeof(float)), cudaMemcpyDeviceToHost));
    for(int i = 0; i<nspins; ++i){
      e4_s = e4_s + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }

    e4_s = e4_s/nspins;
  }
*/

  //CUDA_CALL(cudaUnbindTexture(tex_x_float));
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

  free_dia(J1ij_s_dev);
  free_dia(J1ij_t_dev);
  free_dia(J2ij_s_dev);
  free_dia(J2ij_t_dev);
  free_csr_4d(J4ijkl_s_dev);
}

