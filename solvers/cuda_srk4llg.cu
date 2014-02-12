// Copyright 2014 Joseph Barker. All rights reserved.

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

#include "solvers/cuda_srk4llg.h"
#include "solvers/cuda_srk4llg_kernel.h"

#include "jblib/containers/array.h"

void CUDALLGSolverSRK4::sync_device_data()
{
    using namespace globals;
    CUDA_CALL(cudaMemcpy(s.data(), s_dev, (size_t)(num_spins3*sizeof(double)), cudaMemcpyDeviceToHost));
}

void CUDALLGSolverSRK4::initialize(int argc, char **argv, double idt)
{
    using namespace globals;

    // initialize base class
    Solver::initialize(argc, argv, idt);

    sigma.resize(num_spins);

    for(int i = 0; i<num_spins; ++i) {
        sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)) / (dt*mus(i)*mu_bohr_si) );
    }


    output.write("  * CUDA SRK4 LLG solver (GPU)\n");

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

    //-------------------------------------------------------------------
    //  Allocate device memory
    //-------------------------------------------------------------------

    output.write("  * Converting MAP to DIA\n");
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

    // Allocate double arrays
    CUDA_CALL(cudaMalloc((void**)&s_dev, num_spins3*sizeof(double)));   // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&s_old_dev, num_spins3*sizeof(double)));   // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&k0_dev, num_spins3*sizeof(double)));  // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&k1_dev, num_spins3*sizeof(double)));  // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&k2_dev, num_spins3*sizeof(double)));  // 3*num_spins

    // Allocate float arrays
    CUDA_CALL(cudaMalloc((void**)&sf_dev, num_spins3*sizeof(float)));       // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&h_dev, num_spins3*sizeof(float)));        // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&h_dipole_dev, num_spins3*sizeof(float)));  // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&e_dev, num_spins3*sizeof(float)));        // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&r_dev, num_spins3*sizeof(float)));        // 3*num_spins
    CUDA_CALL(cudaMalloc((void**)&r_max_dev, 3*sizeof(float)));          // 3
    CUDA_CALL(cudaMalloc((void**)&pbc_dev, 3*sizeof(bool)));             // 3
    CUDA_CALL(cudaMalloc((void**)&mat_dev, num_spins*4*sizeof(float)));     // 4*num_spins

    // CURAND requires that the array is a multiple of 2
    CUDA_CALL(cudaMalloc((void**)&w_dev, (num_spins3+(num_spins3%2))*sizeof(float)));  // 3*num_spins (+1 if odd)

    //-------------------------------------------------------------------
    //  Transfer data to device memory
    //-------------------------------------------------------------------

    allocate_transfer_dia(J1ij_s, J1ij_s_dev);
    allocate_transfer_dia(J1ij_t, J1ij_t_dev);
    allocate_transfer_dia(J2ij_s, J2ij_s_dev);
    allocate_transfer_dia(J2ij_t, J2ij_t_dev);
    allocate_transfer_csr_4d(J4ijkl_s, J4ijkl_s_dev);

    //-------------------------------------------------------------------
    //  Copy data to device
    //-------------------------------------------------------------------

    output.write("  * Copying data to device memory...\n");


    // Initial spin configuration
    {
        jblib::Array<float, 2> sf(num_spins, 3);
            for(int i = 0; i<num_spins; ++i) {
                for(int j = 0; j<3; ++j) {
                    sf(i, j) = static_cast<float>(s(i, j));
                }
            }

        CUDA_CALL(cudaMemcpy(s_dev, s.data(), (size_t)(num_spins3*sizeof(double)), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(s_old_dev, s.data(), (size_t)(num_spins3*sizeof(double)), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(sf_dev, sf.data(), (size_t)(num_spins3*sizeof(float)), cudaMemcpyHostToDevice));
    }

    // Lattice dimensions
    {
        float r_maxf[3];
        lattice.getMaxDimensions(r_maxf[0], r_maxf[1], r_maxf[2]);
        CUDA_CALL(cudaMemcpy(r_max_dev, r_maxf, (size_t)(3*sizeof(float)), cudaMemcpyHostToDevice));
    }

    // Periodic boundary conditions
    {
        bool pbc[3];
        lattice.getBoundaries(pbc[0], pbc[1], pbc[2]);
        CUDA_CALL(cudaMemcpy(pbc_dev, pbc, (size_t)(3*sizeof(bool)), cudaMemcpyHostToDevice));
    }

    // Atom positions
    CUDA_CALL(cudaMemcpy(r_dev, atom_pos.data(), (size_t)(num_spins3*sizeof(float)), cudaMemcpyHostToDevice));

    // Material properties
    {
        jblib::Array<float, 2> mat(num_spins, 4);
        for(int i = 0; i<num_spins; ++i){
            mat(i, 0) = mus(i);
            mat(i, 1) = gyro(i);
            mat(i, 2) = alpha(i);
            mat(i, 3) = sigma(i);
        }

        CUDA_CALL(cudaMemcpy(mat_dev, mat.data(), (size_t)(num_spins*4*sizeof(float)), cudaMemcpyHostToDevice));
    }



    //-------------------------------------------------------------------
    //  initialize arrays to zero
    //-------------------------------------------------------------------

    CUDA_CALL(cudaMemset(w_dev, 0, num_spins3*sizeof(float)));
    CUDA_CALL(cudaMemset(h_dev, 0, num_spins3*sizeof(float)));
    CUDA_CALL(cudaMemset(h_dipole_dev, 0, num_spins3*sizeof(float)));
    CUDA_CALL(cudaMemset(e_dev, 0, num_spins3*sizeof(float)));

    //-------------------------------------------------------------------
    //  Determine blocksizes
    //-------------------------------------------------------------------
    nblocks = (num_spins+BLOCKSIZE-1)/BLOCKSIZE;


    J1ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
    J1ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
    J2ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
    J2ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE, (num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
    J4ijkl_s_dev.blocks = std::min<int>(CSR_4D_BLOCK_SIZE, (num_spins+CSR_4D_BLOCK_SIZE-1)/CSR_4D_BLOCK_SIZE);

    eng.resize(num_spins, 3);

    initialized = true;
}

void CUDALLGSolverSRK4::run()
{
  using namespace globals;

  // generate wiener trajectories
  float stmp = sqrt(temperature());

  if(temperature() > 0.0) {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, (num_spins3+(num_spins3%2)), 0.0f, stmp));
  }

    cuda_device_compute_fields(J1ij_s_dev, J1ij_t_dev, J2ij_s_dev, J2ij_t_dev, J4ijkl_s_dev, sf_dev, r_dev, r_max_dev, mat_dev, pbc_dev, h_dev, h_dipole_dev, true);

  // Integrate to find K0
  CUDAIntegrateLLG_SRK4<<<nblocks, BLOCKSIZE>>>
    (s_dev, s_old_dev, k0_dev, h_dev, w_dev, sf_dev, mat_dev, h_app[0], h_app[1], h_app[2], 0.5, dt, num_spins);

    cuda_device_compute_fields(J1ij_s_dev, J1ij_t_dev, J2ij_s_dev, J2ij_t_dev, J4ijkl_s_dev, sf_dev, r_dev, r_max_dev, mat_dev, pbc_dev, h_dev, h_dipole_dev, false);

  // Integrate to find K1
  CUDAIntegrateLLG_SRK4<<<nblocks, BLOCKSIZE>>>
    (s_dev, s_old_dev, k1_dev, h_dev, w_dev, sf_dev, mat_dev, h_app[0], h_app[1], h_app[2], 0.5, dt, num_spins);

    cuda_device_compute_fields(J1ij_s_dev, J1ij_t_dev, J2ij_s_dev, J2ij_t_dev, J4ijkl_s_dev, sf_dev, r_dev, r_max_dev, mat_dev, pbc_dev, h_dev, h_dipole_dev, false);

  // Integrate to find K2
  CUDAIntegrateLLG_SRK4<<<nblocks, BLOCKSIZE>>>
    (s_dev, s_old_dev, k2_dev, h_dev, w_dev, sf_dev, mat_dev, h_app[0], h_app[1], h_app[2], 1.0, dt, num_spins);

    cuda_device_compute_fields(J1ij_s_dev, J1ij_t_dev, J2ij_s_dev, J2ij_t_dev, J4ijkl_s_dev, sf_dev, r_dev, r_max_dev, mat_dev, pbc_dev, h_dev, h_dipole_dev, false);

  // Integrate to find K3
  CUDAIntegrateEndPointLLG_SRK4<<<nblocks, BLOCKSIZE>>>
    (s_dev, s_old_dev, k0_dev, k1_dev, k2_dev, h_dev, w_dev, sf_dev, mat_dev, h_app[0], h_app[1], h_app[2], dt, num_spins);

  iteration++;
}

void CUDALLGSolverSRK4::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s){
}

CUDALLGSolverSRK4::~CUDALLGSolverSRK4()
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

  CUDA_CALL(cudaFree(s_dev));
  CUDA_CALL(cudaFree(s_old_dev));
  CUDA_CALL(cudaFree(k0_dev));
  CUDA_CALL(cudaFree(k1_dev));
  CUDA_CALL(cudaFree(k2_dev));
  CUDA_CALL(cudaFree(sf_dev));
  CUDA_CALL(cudaFree(w_dev));
  CUDA_CALL(cudaFree(r_dev));
  CUDA_CALL(cudaFree(r_max_dev));
  CUDA_CALL(cudaFree(pbc_dev));
  CUDA_CALL(cudaFree(h_dev));
  CUDA_CALL(cudaFree(h_dipole_dev));
  CUDA_CALL(cudaFree(e_dev));
  CUDA_CALL(cudaFree(mat_dev));
}

