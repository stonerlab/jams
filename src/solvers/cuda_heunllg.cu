#include "cuda_sparse.h"
#include "cuda_sparse_types.h"
#include "cuda_heunllg_kernel.h"
#include "globals.h"
#include "consts.h"

#include "cuda_heunllg.h"

#include <curand.h>
#include <cuda.h>
#include <cublas.h>
#include <cusparse.h>

#include <cmath>


void CUDAHeunLLGSolver::syncOutput()
{
  using namespace globals;
  CUDA_CALL(cudaMemcpy(s.ptr(),s_dev,(size_t)(nspins3*sizeof(double)),cudaMemcpyDeviceToHost));
}

void CUDAHeunLLGSolver::initialise(int argc, char **argv, double idt)
{
  using namespace globals;

  // initialise base class
  Solver::initialise(argc,argv,idt);

  sigma.resize(nspins);

  for(int i=0; i<nspins; ++i) {
    sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)) / (dt*mus(i)*mu_bohr_si) );
  }


  output.write("  * CUDA Heun LLG solver (GPU)\n");

  //-------------------------------------------------------------------
  //  Initialise curand
  //-------------------------------------------------------------------

  output.write("  * Initialising CURAND...\n");
  // curand generator
  CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));


  // TODO: set random seed from config
  const unsigned long long gpuseed = rng.uniform()*18446744073709551615ULL;
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, gpuseed));
  CURAND_CALL(curandGenerateSeeds(gen));
  /*CUDA_CALL(cudaThreadSetLimit(cudaLimitStackSize,1024));*/
  /*CUDA_CALL(cudaThreadSynchronize());*/


  //-------------------------------------------------------------------
  //  Allocate device memory
  //-------------------------------------------------------------------

  output.write("  * Converting MAP to DIA\n");
  J1ij_s.convertMAP2DIA();
  J1ij_t.convertMAP2DIA();
  J2ij_s.convertMAP2DIA();
  J2ij_t.convertMAP2DIA();
  output.write("  * J1ij scalar matrix memory (DIA): %f MB\n",J1ij_s.calculateMemory());
  output.write("  * J1ij tensor matrix memory (DIA): %f MB\n",J1ij_t.calculateMemory());
  output.write("  * J2ij scalar matrix memory (DIA): %f MB\n",J2ij_s.calculateMemory());
  output.write("  * J2ij tensor matrix memory (DIA): %f MB\n",J2ij_t.calculateMemory());
  
  output.write("  * Converting J4 MAP to CSR\n");
  J4ijkl_s.convertMAP2CSR();
  output.write("  * J4ijkl scalar matrix memory (CSR): %f MB\n",J4ijkl_s.calculateMemory());


  output.write("  * Allocating device memory...\n");
  // spin arrays
  CUDA_CALL(cudaMalloc((void**)&s_dev,nspins3*sizeof(double)));
  CUDA_CALL(cudaMalloc((void**)&sf_dev,nspins3*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&s_new_dev,nspins3*sizeof(double)));

  // field arrays
  CUDA_CALL(cudaMalloc((void**)&h_dev,nspins3*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&e_dev,nspins3*sizeof(float)));

  // position arrays
  CUDA_CALL(cudaMalloc((void**)&r_dev,nspins3*sizeof(float)));

  if(nspins3%2 == 0) {
    // wiener processes
    CUDA_CALL(cudaMalloc((void**)&w_dev,nspins3*sizeof(float)));
  } else {
    CUDA_CALL(cudaMalloc((void**)&w_dev,(nspins3+1)*sizeof(float)));
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
  CUDA_CALL(cudaMalloc((void**)&mat_dev,nspins*4*sizeof(float)));

  //-------------------------------------------------------------------
  //  Copy data to device
  //-------------------------------------------------------------------

  output.write("  * Copying data to device memory...\n");
  // initial spins
  Array2D<float> sf(nspins,3);
  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sf(i,j) = static_cast<float>(s(i,j));
    }
  }
  CUDA_CALL(cudaMemcpy(s_dev,s.ptr(),(size_t)(nspins3*sizeof(double)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(sf_dev,sf.ptr(),(size_t)(nspins3*sizeof(float)),cudaMemcpyHostToDevice));

  // position array
  CUDA_CALL(cudaMemcpy(r_dev,r_pos.ptr(),(size_t)(nspins3*sizeof(float)),cudaMemcpyHostToDevice));

  Array2D<float> mat(nspins,4);
  // material properties
  for(int i=0; i<nspins; ++i){
    mat(i,0) = mus(i);
    mat(i,1) = gyro(i);
    mat(i,2) = alpha(i);
    mat(i,3) = sigma(i);
  }
  CUDA_CALL(cudaMemcpy(mat_dev,mat.ptr(),(size_t)(nspins*4*sizeof(float)),cudaMemcpyHostToDevice));

  eng.resize(nspins,3);


  //-------------------------------------------------------------------
  //  Initialise arrays to zero
  //-------------------------------------------------------------------
  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sf(i,j) = 0.0;
    }
  }
  
  CUDA_CALL(cudaMemcpy(w_dev,sf.ptr(),(size_t)(nspins3*sizeof(float)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(h_dev,sf.ptr(),(size_t)(nspins3*sizeof(float)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(e_dev,sf.ptr(),(size_t)(nspins3*sizeof(float)),cudaMemcpyHostToDevice));

  nblocks = (nspins+BLOCKSIZE-1)/BLOCKSIZE;

  J1ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE,(nspins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
  J1ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE,(nspins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

  J2ij_s_dev.blocks = std::min<int>(DIA_BLOCK_SIZE,(nspins+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
  J2ij_t_dev.blocks = std::min<int>(DIA_BLOCK_SIZE,(nspins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
  
  J4ijkl_s_dev.blocks = std::min<int>(CSR_4D_BLOCK_SIZE,(nspins+CSR_4D_BLOCK_SIZE-1)/CSR_4D_BLOCK_SIZE);

  initialised = true;
}

void CUDAHeunLLGSolver::run()
{
  using namespace globals;

  // generate wiener trajectories
  float stmp = sqrt(temperature);
  
  if(temperature > 0.0) {
    if(nspins3%2 == 0) {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, nspins3, 0.0f, stmp));
    } else {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, (nspins3+1), 0.0f, stmp));
    }
  }
  
  // calculate interaction fields (and zero field array)

  //CUDA_CALL(cudaBindTexture(0,tex_x_float,sf_dev));
  
  float beta=0;
  // bilinear scalar
  if(J1ij_s.nonZero() > 0){
    bilinear_scalar_dia_kernel<<< J1ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(nspins,nspins,
      J1ij_s.diags(),J1ij_s_dev.pitch,1.0,beta,J1ij_s_dev.row,J1ij_s_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }

  // bilinear tensor
  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(nspins3,nspins3,
      J1ij_t.diags(),J1ij_t_dev.pitch,beta,1.0,J1ij_t_dev.row,J1ij_t_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }
  
  // biquadratic scalar
  if(J2ij_s.nonZero() > 0){
    biquadratic_scalar_dia_kernel<<< J2ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(nspins,nspins,
      J2ij_s.diags(),J2ij_s_dev.pitch,2.0,beta,J2ij_s_dev.row,J2ij_s_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }
  
  // biquadratic tensor
  if(J2ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J2ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(nspins3,nspins3,
      J2ij_t.diags(),J2ij_t_dev.pitch,2.0,beta,J2ij_t_dev.row,J2ij_t_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }
  
  if(J4ijkl_s.nonZero() > 0){
    fourspin_scalar_csr_kernel<<< J4ijkl_s_dev.blocks,CSR_4D_BLOCK_SIZE>>>(nspins,nspins,1.0,beta,
        J4ijkl_s_dev.pointers,J4ijkl_s_dev.coords,J4ijkl_s_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }
  
  //CUDA_CALL(cudaUnbindTexture(tex_x_float));
  
  // integrate
  cuda_heun_llg_kernelA<<<nblocks,BLOCKSIZE>>>
    (
      s_dev,
      sf_dev,
      s_new_dev,
      h_dev,
      w_dev,
      mat_dev,
      h_app[0],
      h_app[1],
      h_app[2],
      nspins,
      dt
    );

  // calculate interaction fields (and zero field array)

  beta=0.0;
  // bilinear scalar
  if(J1ij_s.nonZero() > 0){
    bilinear_scalar_dia_kernel<<< J1ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(nspins,nspins,
      J1ij_s.diags(),J1ij_s_dev.pitch,1.0,beta,J1ij_s_dev.row,J1ij_s_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }

  // bilinear tensor
  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(nspins3,nspins3,
      J1ij_t.diags(),J1ij_t_dev.pitch,1.0,beta,J1ij_t_dev.row,J1ij_t_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }
  
  // biquadratic scalar
  if(J2ij_s.nonZero() > 0){
    biquadratic_scalar_dia_kernel<<< J2ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(nspins,nspins,
      J2ij_s.diags(),J2ij_s_dev.pitch,2.0,beta,J2ij_s_dev.row,J2ij_s_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }
  
  // biquadratic tensor
  if(J2ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J2ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(nspins3,nspins3,
      J2ij_t.diags(),J2ij_t_dev.pitch,2.0,beta,J2ij_t_dev.row,J2ij_t_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }
  
  if(J4ijkl_s.nonZero() > 0){
    fourspin_scalar_csr_kernel<<< J4ijkl_s_dev.blocks,CSR_4D_BLOCK_SIZE>>>(nspins,nspins,1.0,beta,
        J4ijkl_s_dev.pointers,J4ijkl_s_dev.coords,J4ijkl_s_dev.val,sf_dev,h_dev);
    beta = 1.0;
  }

  /*Array2D<float> hf(nspins,3);*/
  /*Array2D<float> sf(nspins,3);*/
  /*CUDA_CALL(cudaMemcpy(hf.ptr(),h_dev,(size_t)(nspins3*sizeof(float)),cudaMemcpyDeviceToHost));*/
  /*CUDA_CALL(cudaMemcpy(sf.ptr(),sf_dev,(size_t)(nspins3*sizeof(float)),cudaMemcpyDeviceToHost));*/

  /*for(int i=0; i<nspins; ++i){*/
      /*std::cout<<i<<sf(i,0)<<"\t"<<sf(i,1)<<"\t"<<sf(i,2)<<"\t"<<hf(i,0)<<"\t"<<hf(i,1)<<"\t"<<hf(i,2)<<std::endl;*/
  /*}*/
  
  //CUDA_CALL(cudaUnbindTexture(tex_x_float));
  
  cuda_heun_llg_kernelB<<<nblocks,BLOCKSIZE>>>
    (
      s_dev,
      sf_dev,
      s_new_dev,
      h_dev,
      w_dev,
      mat_dev,
      h_app[0],
      h_app[1],
      h_app[2],
      nspins,
      dt
    );
  iteration++;
}

void CUDAHeunLLGSolver::calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s){
  using namespace globals;
  const float beta=0.0;

  e1_s = 0.0; e1_t = 0.0; e2_s = 0.0; e2_t = 0.0;
  
  //size_t offset = size_t(-1);
  //CUDA_CALL(cudaBindTexture(&offset,tex_x_float,sf_dev));

  // bilinear scalar
  if(J1ij_s.nonZero() > 0){
    bilinear_scalar_dia_kernel<<< J1ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(nspins,nspins,
      J1ij_s.diags(),J1ij_s_dev.pitch,1.0,beta,J1ij_s_dev.row,J1ij_s_dev.val,sf_dev,e_dev);
    CUDA_CALL(cudaMemcpy(eng.ptr(),e_dev,(size_t)(nspins3*sizeof(float)),cudaMemcpyDeviceToHost));
    for(int i=0; i<nspins; ++i){
      e1_s = e1_s + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    e1_s = e1_s/nspins;
  }


  // bilinear tensor
  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J1ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(nspins3,nspins3,
      J1ij_t.diags(),J1ij_t_dev.pitch,1.0,beta,J1ij_t_dev.row,J1ij_t_dev.val,sf_dev,e_dev);
    CUDA_CALL(cudaMemcpy(eng.ptr(),e_dev,(size_t)(nspins3*sizeof(float)),cudaMemcpyDeviceToHost));
    for(int i=0; i<nspins; ++i){
      e1_t = e1_t + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    e1_t = e1_t/nspins;
  }

  
  // biquadratic scalar
  if(J2ij_s.nonZero() > 0){
    biquadratic_scalar_dia_kernel<<< J2ij_s_dev.blocks, DIA_BLOCK_SIZE >>>(nspins,nspins,
      J2ij_s.diags(),J2ij_s_dev.pitch,1.0,beta,J2ij_s_dev.row,J2ij_s_dev.val,sf_dev,e_dev);
    CUDA_CALL(cudaMemcpy(eng.ptr(),e_dev,(size_t)(nspins3*sizeof(float)),cudaMemcpyDeviceToHost));
    for(int i=0; i<nspins; ++i){
      e2_s = e2_s + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    
    e2_s = e2_s/nspins;
  }

  // biquadratic tensor
  if(J2ij_t.nonZero() > 0){
    spmv_dia_kernel<<< J2ij_t_dev.blocks, DIA_BLOCK_SIZE >>>(nspins3,nspins3,
      J2ij_t.diags(),J2ij_t_dev.pitch,1.0,beta,J2ij_t_dev.row,J2ij_t_dev.val,sf_dev,e_dev);
    CUDA_CALL(cudaMemcpy(eng.ptr(),e_dev,(size_t)(nspins3*sizeof(float)),cudaMemcpyDeviceToHost));

    for(int i=0; i<nspins; ++i){
      e2_t = e2_t + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    
    e2_t = e2_t/nspins;
  }
  
  if(J4ijkl_s.nonZero() > 0){
    fourspin_scalar_csr_kernel<<< J4ijkl_s_dev.blocks,CSR_4D_BLOCK_SIZE>>>(nspins,nspins,1.0,beta,
        J4ijkl_s_dev.pointers,J4ijkl_s_dev.coords,J4ijkl_s_dev.val,sf_dev,e_dev);
    CUDA_CALL(cudaMemcpy(eng.ptr(),e_dev,(size_t)(nspins3*sizeof(float)),cudaMemcpyDeviceToHost));
    for(int i=0; i<nspins; ++i){
      e4_s = e4_s + (s(i,0)*eng(i,0)+s(i,1)*eng(i,1)+s(i,2)*eng(i,2));
    }
    
    e4_s = e4_s/nspins;
  }
  
  
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

  // spin arrays
  CUDA_CALL(cudaFree(s_dev));
  CUDA_CALL(cudaFree(sf_dev));
  CUDA_CALL(cudaFree(s_new_dev));

  // field arrays
  CUDA_CALL(cudaFree(r_dev));
  CUDA_CALL(cudaFree(h_dev));
  CUDA_CALL(cudaFree(e_dev));

  // wiener processes
  CUDA_CALL(cudaFree(w_dev));


  // material arrays
  CUDA_CALL(cudaFree(mat_dev));
}

