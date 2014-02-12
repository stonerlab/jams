// Copyright 2014 Joseph Barker. All rights reserved.

#ifdef CUDA

#include "solvers/cuda_heunllbp.h"

#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>

#include "solvers/cuda_heunllbp_kernel.h"

#include "core/consts.h"
#include "core/cuda_sparse.h"
#include "core/cuda_sparse_types.h"
#include "core/globals.h"

#include "jblib/containers/array.h"

void CUDAHeunLLBPSolver::sync_device_data()
{
  using namespace globals;
  CUDA_CALL(cudaMemcpy(s.data(), s_dev, (size_t)(num_spins3*sizeof(double)), cudaMemcpyDeviceToHost));
}

void CUDAHeunLLBPSolver::initialize(int argc, char **argv, double idt)
{
	using namespace globals;

  // initialize base class
	Solver::initialize(argc, argv, idt);

	output.write("  * CUDA Heun LLBP solver (GPU)\n");


	libconfig::Setting &matcfg = config.lookup("materials");

	t_corr.resize(num_spins, 2);
  // read correlation times
	for(int i = 0; i<num_spins; ++i){
		int type_num = lattice.getType(i);
		for(int j = 0; j<2; ++j){
			t_corr(i, j) = matcfg[type_num]["t_corr"][j];
			t_corr(i, j) = gamma_electron_si*t_corr(i, j);
		}

	// calculate sigma for Wiener processes
		sigma.resize(num_spins);
		sigma(i) =  (t_corr(i, 0)/(t_corr(i, 0)-t_corr(i, 1)))*sqrt( (2.0*boltzmann_si*alpha(i)) / (dt*mus(i)*mu_bohr_si));

	// read gyroscopic factors
		gyro(i) = matcfg[type_num]["gyro"];
		gyro(i) = -gyro(i)/mus(i);

	}

  //-------------------------------------------------------------------
  //  initialize curand
  //-------------------------------------------------------------------

	output.write("  * Initialising CURAND...\n");

	const unsigned long long gpuseed = rng.uniform()*18446744073709551615ULL;
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, gpuseed));
	CURAND_CALL(curandGenerateSeeds(gen));

	output.write("  * Allocating Wiener process array...\n");

	if(num_spins3%2 == 0) {
		CUDA_CALL(cudaMalloc((void**)&w_dev, num_spins3*sizeof(float)));
	} else {
		CUDA_CALL(cudaMalloc((void**)&w_dev, (num_spins3+1)*sizeof(float)));
	}

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

	output.write("  * Allocating device interaction matrices...\n");

  // bilinear scalar
	allocate_transfer_dia(J1ij_s, J1ij_s_dev);

  // bilinear tensor
	allocate_transfer_dia(J1ij_t, J1ij_t_dev);

  // biquadratic scalar
	allocate_transfer_dia(J2ij_s, J2ij_s_dev);

  // bilinear tensor
	allocate_transfer_dia(J2ij_t, J2ij_t_dev);

	allocate_transfer_csr_4d(J4ijkl_s, J4ijkl_s_dev);


  //-------------------------------------------------------------------
  //  initialize Spin Arrays
  //-------------------------------------------------------------------

	output.write("  * Allocating spin arrays in device memory...\n");

  // initial spins
	jblib::Array<float, 2> sf(num_spins, 3);
	for(int i = 0; i<num_spins; ++i) {
		for(int j = 0; j<3; ++j) {
			sf(i, j) = static_cast<float>(s(i, j));
		}
	}

	output.write("  * Allocating spin arrays in device memory...\n");

	CUDA_CALL(cudaMalloc((void**)&s_dev, num_spins3*sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&sf_dev, num_spins3*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&s_new_dev, num_spins3*sizeof(double)));

	output.write("  * Copying spin arrays to device memory...\n");

	CUDA_CALL(cudaMemcpy(s_dev, s.data(), (size_t)(num_spins3*sizeof(double)), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(sf_dev, sf.data(), (size_t)(num_spins3*sizeof(float)), cudaMemcpyHostToDevice));


  //-------------------------------------------------------------------
  //  initialize Colouring Processes
  //-------------------------------------------------------------------

  // initialize colouring processes
	jblib::Array<double, 2> u1(num_spins, 3);
	jblib::Array<double, 2> u2(num_spins, 3);


	for(int i = 0; i<num_spins; ++i){
		for(int j = 0; j<3; ++j){
			const double w10 = rng.normal();
			const double w20 = rng.normal();
			u1(i, j) = sqrt(dt)*sigma(i)*(1.0/sqrt(t_corr(i, 0)))*w10;
			u2(i, j) = sqrt(dt)*sigma(i)*(1.0/sqrt(t_corr(i, 1)))*
				( ( (2*sqrt(t_corr(i, 0)*t_corr(i, 1))) / (t_corr(i, 0)+t_corr(i, 1)) )*w10 +
					((t_corr(i, 0)-t_corr(i, 1))/(t_corr(i, 0)+t_corr(i, 1)))*w20);
		}
	}

	output.write("  * Allocating colour processes in device memory...\n");

	CUDA_CALL(cudaMalloc((void**)&u1_dev, num_spins3*sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&u2_dev, num_spins3*sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&u1_new_dev, num_spins3*sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&u2_new_dev, num_spins3*sizeof(double)));

	output.write("  * Copying colour processes to device memory...\n");

	CUDA_CALL(cudaMemcpy(u1_dev, u1.data(), (size_t)(num_spins3*sizeof(double)), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(u2_dev, u2.data(), (size_t)(num_spins3*sizeof(double)), cudaMemcpyHostToDevice));

  //-------------------------------------------------------------------
  //  initialize Correlation Time Array
  //-------------------------------------------------------------------

	output.write("  * Allocating correlation time array in device memory...\n");
	CUDA_CALL(cudaMalloc((void**)&tc_dev, num_spins*2*sizeof(float)));

	output.write("  * Copying correlation time array to device memory...\n");
	CUDA_CALL(cudaMemcpy(tc_dev, t_corr.data(), (size_t)(num_spins*2*sizeof(float)), cudaMemcpyHostToDevice));

  //-------------------------------------------------------------------
  //  initialize Material Array
  //-------------------------------------------------------------------

	jblib::Array<float, 2> mat(num_spins, 4);
	for(int i = 0; i<num_spins; ++i){
		mat(i, 0) = mus(i);
		mat(i, 1) = gyro(i);
		mat(i, 2) = alpha(i);
		mat(i, 3) = sigma(i);
	}

	output.write("  * Allocating material property array in device memory...\n");
	CUDA_CALL(cudaMalloc((void**)&mat_dev, num_spins*4*sizeof(float)));

	output.write("  * Copying material property array to device memory...\n");
	CUDA_CALL(cudaMemcpy(mat_dev, mat.data(), (size_t)(num_spins*4*sizeof(float)), cudaMemcpyHostToDevice));


  //-------------------------------------------------------------------
  //  initialize arrays to zero
  //-------------------------------------------------------------------
	jblib::Array<float, 2> hf(num_spins, 3);

	for(int i = 0; i<num_spins; ++i) {
		for(int j = 0; j<3; ++j) {
			hf(i, j) = 0.0;
		}
	}

	output.write("  * Allocating field array in device memory...\n");
	CUDA_CALL(cudaMalloc((void**)&h_dev, num_spins3*sizeof(float)));

	output.write("  * Copying field array to device memory...\n");
	CUDA_CALL(cudaMemcpy(h_dev, hf.data(), (size_t)(num_spins3*sizeof(float)), cudaMemcpyHostToDevice));



	nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;

	J1ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
	J1ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

	J2ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
	J2ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

	J4ijkl_s_dev.blocks = std::min<int>(CSR_4D_BLOCK_SIZE, (num_spins+CSR_4D_BLOCK_SIZE-1)/CSR_4D_BLOCK_SIZE);

	initialized = true;
}

void CUDAHeunLLBPSolver::run()
{
  using namespace globals;

  // generate wiener trajectories
  float stmp = sqrt(temperature());

  if(temperature() > 0.0) {
    if(num_spins3%2 == 0) {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, num_spins3, 0.0f, stmp));
    } else {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, (num_spins3+1), 0.0f, stmp));
    }
  }


  // calculate interaction fields (and zero field array)

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

  // integrate
  cuda_heun_llbp_kernelA<<<nblocks, BLOCKSIZE>>>
    (
      s_dev,
      sf_dev,
      s_new_dev,
      h_dev,
      w_dev,
	  u1_dev,
	  u1_new_dev,
	  u2_dev,
	  u2_new_dev,
	  tc_dev,
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

  cuda_heun_llbp_kernelB<<<nblocks, BLOCKSIZE>>>
    (
        s_dev,
        sf_dev,
        s_new_dev,
        h_dev,
        w_dev,
  	  u1_dev,
  	  u1_new_dev,
  	  u2_dev,
  	  u2_new_dev,
  	  tc_dev,
        mat_dev,
        h_app[0],
        h_app[1],
        h_app[2],
        num_spins,
        dt

    );
  iteration++;
}

void CUDAHeunLLBPSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s){
  using namespace globals;

}

CUDAHeunLLBPSolver::~CUDAHeunLLBPSolver()
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

  CUDA_CALL(cudaFree(u1_dev));
  CUDA_CALL(cudaFree(u1_new_dev));
  CUDA_CALL(cudaFree(u2_dev));
  CUDA_CALL(cudaFree(u2_new_dev));
  CUDA_CALL(cudaFree(tc_dev));

  // field arrays
  CUDA_CALL(cudaFree(h_dev));

  // wiener processes
  CUDA_CALL(cudaFree(w_dev));


  // material arrays
  CUDA_CALL(cudaFree(mat_dev));
}

#endif
