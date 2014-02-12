// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/cuda_heunllms.h"

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

#include "solvers/cuda_heunllms_kernel.h"

#include "jblib/containers/array.h"

void CUDAHeunLLMSSolver::sync_device_data()
{
  using namespace globals;
  CUDA_CALL(cudaMemcpy(s.data(), s_dev, (size_t)(num_spins3*sizeof(double)), cudaMemcpyDeviceToHost));
}

void CUDAHeunLLMSSolver::initialize(int argc, char **argv, double idt)
{
  using namespace globals;

  // initialize base class
  Solver::initialize(argc, argv, idt);

  output.write("  * CUDA Heun LLMS solver (GPU)\n");

  sigma.resize(num_spins);

  libconfig::Setting &matcfg = config.lookup("materials");

  for(int i = 0; i<num_spins; ++i){
    int type_num = lattice.getType(i);
	omega_corr(i) = matcfg[type_num]["t_corr"];
    omega_corr(i) = 1.0/(gamma_electron_si*omega_corr(i));

    sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)*omega_corr(i)*omega_corr(i)) / (dt*mus(i)*mu_bohr_si) );

  	gyro(i) = matcfg[type_num]["gyro"];
	gyro(i) = -gyro(i)/mus(i);
  }


  //-------------------------------------------------------------------
  //  initialize curand
  //-------------------------------------------------------------------

  output.write("  * Initialising CURAND...\n");
  // curand generator
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));


  // TODO: set random seed from config
  const unsigned long long gpuseed = rng.uniform()*18446744073709551615ULL;
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, gpuseed));
  CURAND_CALL(curandGenerateSeeds(gen));
  /*CUDA_CALL(cudaThreadSetLimit(cudaLimitStackSize, 1024));*/
  /*CUDA_CALL(cudaThreadSynchronize());*/


  //-------------------------------------------------------------------
  //  Allocate device memory
  //-------------------------------------------------------------------

  output.write("  * Converting MAP to DIA\n");
  J1ij_s.convertMAP2DIA();
  J1ij_t.convertMAP2DIA();
  J2ij_s.convertMAP2DIA();
  J2ij_t.convertMAP2DIA();
  output.write("  * J1ij scalar matrix memory (DIA): %f MB\n", J1ij_s.calculateMemory());
  output.write("  * J1ij tensor matrix memory (DIA): %f MB\n", J1ij_t.calculateMemory());
  output.write("  * J2ij scalar matrix memory (DIA): %f MB\n", J2ij_s.calculateMemory());
  output.write("  * J2ij tensor matrix memory (DIA): %f MB\n", J2ij_t.calculateMemory());

  output.write("  * Converting J4 MAP to CSR\n");
  /*J4ijkl_s.convertMAP2CSR();*/
  output.write("  * J4ijkl scalar matrix memory (CSR): %f MB\n", J4ijkl_s.calculateMemoryUsage());


  output.write("  * Allocating device memory...\n");
  // spin arrays
  CUDA_CALL(cudaMalloc((void**)&s_dev, num_spins3*sizeof(double)));
  CUDA_CALL(cudaMalloc((void**)&sf_dev, num_spins3*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&s_new_dev, num_spins3*sizeof(double)));

  // stochastic process arrays
  CUDA_CALL(cudaMalloc((void**)&u_dev, num_spins3*sizeof(double)));
  CUDA_CALL(cudaMalloc((void**)&u_new_dev, num_spins3*sizeof(double)));
  CUDA_CALL(cudaMalloc((void**)&omega_corr_dev, num_spins*sizeof(float)));

  // field arrays
  CUDA_CALL(cudaMalloc((void**)&h_dev, num_spins3*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&e_dev, num_spins3*sizeof(float)));

  if(num_spins3%2 == 0) {
    // wiener processes
    CUDA_CALL(cudaMalloc((void**)&w_dev, num_spins3*sizeof(float)));
  } else {
    CUDA_CALL(cudaMalloc((void**)&w_dev, (num_spins3+1)*sizeof(float)));
  }


  // bilinear scalar
  allocate_transfer_dia(J1ij_s, J1ij_s_dev);

  // bilinear tensor
  allocate_transfer_dia(J1ij_t, J1ij_t_dev);

  // biquadratic scalar
  allocate_transfer_dia(J2ij_s, J2ij_s_dev);

  // bilinear tensor
  allocate_transfer_dia(J2ij_t, J2ij_t_dev);

  allocate_transfer_csr_4d(J4ijkl_s, J4ijkl_s_dev);

  // material properties
  CUDA_CALL(cudaMalloc((void**)&mat_dev, num_spins*4*sizeof(float)));

  //-------------------------------------------------------------------
  //  Copy data to device
  //-------------------------------------------------------------------

  output.write("  * Copying data to device memory...\n");
  // initial spins
  jblib::Array<float, 2> sf(num_spins, 3);
  for(int i = 0; i<num_spins; ++i) {
    for(int j = 0; j<3; ++j) {
      sf(i, j) = static_cast<float>(s(i, j));
    }
  }

  CUDA_CALL(cudaMemcpy(s_dev, s.data(), (size_t)(num_spins3*sizeof(double)), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(sf_dev, sf.data(), (size_t)(num_spins3*sizeof(float)), cudaMemcpyHostToDevice));

  jblib::Array<float, 2> mat(num_spins, 4);
  // material properties
  for(int i = 0; i<num_spins; ++i){
    mat(i, 0) = mus(i);
    mat(i, 1) = gyro(i);
    mat(i, 2) = alpha(i);
    mat(i, 3) = sigma(i);
  }

  CUDA_CALL(cudaMemcpy(mat_dev, mat.data(), (size_t)(num_spins*4*sizeof(float)), cudaMemcpyHostToDevice));

  eng.resize(num_spins, 3);


  //-------------------------------------------------------------------
  //  initialize arrays to zero
  //-------------------------------------------------------------------
  for(int i = 0; i<num_spins; ++i) {
    for(int j = 0; j<3; ++j) {
      sf(i, j) = 0.0;
    }
  }

  CUDA_CALL(cudaMemcpy(w_dev, sf.data(), (size_t)(num_spins3*sizeof(float)), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(h_dev, sf.data(), (size_t)(num_spins3*sizeof(float)), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(e_dev, sf.data(), (size_t)(num_spins3*sizeof(float)), cudaMemcpyHostToDevice));

  jblib::Array<float, 1> tmp(num_spins);
  for(int i = 0; i<num_spins; ++i) {
  	tmp(i) = omega_corr(i);
  }
  CUDA_CALL(cudaMemcpy(omega_corr_dev, tmp.data(), (size_t)(num_spins*sizeof(float)), cudaMemcpyHostToDevice));

  jblib::Array<double, 2> u(num_spins, 3);
  for(int i = 0; i<num_spins; ++i) {
    for(int j = 0; j<3; ++j) {
      u(i, j) = 0.0;
    }
  }
  CUDA_CALL(cudaMemcpy(u_dev, u.data(), (size_t)(num_spins3*sizeof(double)), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(u_new_dev, u.data(), (size_t)(num_spins3*sizeof(double)), cudaMemcpyHostToDevice));


  nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;

  J1ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
  J1ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

  J2ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
  J2ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

  J4ijkl_s_dev.blocks = std::min<int>(CSR_4D_BLOCK_SIZE, (num_spins+CSR_4D_BLOCK_SIZE-1)/CSR_4D_BLOCK_SIZE);

  initialized = true;
}

void CUDAHeunLLMSSolver::run()
{
  using namespace globals;

  // generate wiener trajectories
  float stmp = sqrt(temperature);

  if(temperature > 0.0) {
    if(num_spins3%2 == 0) {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, num_spins3, 0.0f, stmp));
    } else {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, (num_spins3+1), 0.0f, stmp));
    }
  }

  jblib::Array<float, 2> tmp(num_spins, 3);


  // calculate interaction fields (and zero field array)

  //CUDA_CALL(cudaBindTexture(0, tex_x_float, sf_dev));

  float beta=0;
  // bilinear scalar
  if(J1ij_s.nonZero() > 0){
    bilinear_scalar_interaction_dia_kernel<<< J1ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(num_spins, num_spins,
      J1ij_s.diags(), J1ij_s_dev.pitch, 1.0, beta, J1ij_s_dev.row, J1ij_s_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  // bilinear tensor
  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(num_spins3, num_spins3,
      J1ij_t.diags(), J1ij_t_dev.pitch, beta, 1.0, J1ij_t_dev.row, J1ij_t_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  // biquadratic scalar
  if(J2ij_s.nonZero() > 0){
    biquadratic_scalar_dia_kernel<<< J2ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(num_spins, num_spins,
      J2ij_s.diags(), J2ij_s_dev.pitch, 2.0, beta, J2ij_s_dev.row, J2ij_s_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  // biquadratic tensor
  if(J2ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J2ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(num_spins3, num_spins3,
      J2ij_t.diags(), J2ij_t_dev.pitch, 2.0, beta, J2ij_t_dev.row, J2ij_t_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  if(J4ijkl_s.nonZeros() > 0){
    fourspin_scalar_interaction_csr_kernel<<< J4ijkl_s_dev.blocks, CSR_4D_BLOCK_SIZE>>>(num_spins, num_spins, 1.0, beta,
        J4ijkl_s_dev.pointers, J4ijkl_s_dev.coords, J4ijkl_s_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  //CUDA_CALL(cudaUnbindTexture(tex_x_float));

  // integrate
  cuda_heun_llms_kernelA<<<nblocks, BLOCKSIZE>>>
    (
      s_dev,
      sf_dev,
      s_new_dev,
      h_dev,
      w_dev,
	  u_dev,
	  u_new_dev,
	  omega_corr_dev,
      mat_dev,
      h_app[0],
      h_app[1],
      h_app[2],
      num_spins,
      dt
    );
	//   Array2D<float> hf(num_spins, 3);
	//   Array2D<float> sf(num_spins, 3);
	//   CUDA_CALL(cudaMemcpy(hf.data(), h_dev, (size_t)(num_spins3*sizeof(float)), cudaMemcpyDeviceToHost));
	// CUDA_CALL(cudaMemcpy(sf.data(), sf_dev, (size_t)(num_spins3*sizeof(float)), cudaMemcpyDeviceToHost));
	//
	//   for(int i = 0; i<num_spins; ++i){
	//       std::cout<<i<<"\t"<<sf(i, 0)<<"\t"<<sf(i, 1)<<"\t"<<sf(i, 2)<<"\t"<<hf(i, 0)<<"\t"<<hf(i, 1)<<"\t"<<hf(i, 2)<<std::endl;
	//   }


  // calculate interaction fields (and zero field array)

  beta=0.0;
  // bilinear scalar
  if(J1ij_s.nonZero() > 0){
    bilinear_scalar_interaction_dia_kernel<<< J1ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(num_spins, num_spins,
      J1ij_s.diags(), J1ij_s_dev.pitch, 1.0, beta, J1ij_s_dev.row, J1ij_s_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  // bilinear tensor
  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(num_spins3, num_spins3,
      J1ij_t.diags(), J1ij_t_dev.pitch, 1.0, beta, J1ij_t_dev.row, J1ij_t_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  // biquadratic scalar
  if(J2ij_s.nonZero() > 0){
    biquadratic_scalar_dia_kernel<<< J2ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(num_spins, num_spins,
      J2ij_s.diags(), J2ij_s_dev.pitch, 2.0, beta, J2ij_s_dev.row, J2ij_s_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  // biquadratic tensor
  if(J2ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J2ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(num_spins3, num_spins3,
      J2ij_t.diags(), J2ij_t_dev.pitch, 2.0, beta, J2ij_t_dev.row, J2ij_t_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  if(J4ijkl_s.nonZeros() > 0){
    fourspin_scalar_interaction_csr_kernel<<< J4ijkl_s_dev.blocks, CSR_4D_BLOCK_SIZE>>>(num_spins, num_spins, 1.0, beta,
        J4ijkl_s_dev.pointers, J4ijkl_s_dev.coords, J4ijkl_s_dev.val, sf_dev, h_dev);
    beta = 1.0;
  }

  /*Array2D<float> hf(num_spins, 3);*/
  /*Array2D<float> sf(num_spins, 3);*/
  /*CUDA_CALL(cudaMemcpy(hf.data(), h_dev, (size_t)(num_spins3*sizeof(float)), cudaMemcpyDeviceToHost));*/
  /*CUDA_CALL(cudaMemcpy(sf.data(), sf_dev, (size_t)(num_spins3*sizeof(float)), cudaMemcpyDeviceToHost));*/

  /*for(int i = 0; i<num_spins; ++i){*/
      /*std::cout<<i<<sf(i, 0)<<"\t"<<sf(i, 1)<<"\t"<<sf(i, 2)<<"\t"<<hf(i, 0)<<"\t"<<hf(i, 1)<<"\t"<<hf(i, 2)<<std::endl;*/
  /*}*/

  //CUDA_CALL(cudaUnbindTexture(tex_x_float));

  cuda_heun_llms_kernelB<<<nblocks, BLOCKSIZE>>>
    (
      s_dev,
      sf_dev,
      s_new_dev,
      h_dev,
      w_dev,
	  u_dev,
	  u_new_dev,
	  omega_corr_dev,
      mat_dev,
      h_app[0],
      h_app[1],
      h_app[2],
      num_spins,
      dt
    );
  iteration++;
}

void CUDAHeunLLMSSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s){
  using namespace globals;

}

CUDAHeunLLMSSolver::~CUDAHeunLLMSSolver()
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

  // spin arrays
  CUDA_CALL(cudaFree(s_dev));
  CUDA_CALL(cudaFree(sf_dev));
  CUDA_CALL(cudaFree(s_new_dev));

  CUDA_CALL(cudaFree(u_dev));
  CUDA_CALL(cudaFree(u_new_dev));
  CUDA_CALL(cudaFree(omega_corr_dev));

  // field arrays
  CUDA_CALL(cudaFree(h_dev));
  CUDA_CALL(cudaFree(e_dev));

  // wiener processes
  CUDA_CALL(cudaFree(w_dev));


  // material arrays
  CUDA_CALL(cudaFree(mat_dev));
}

