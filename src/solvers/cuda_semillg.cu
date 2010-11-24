#include "cuda_semillg_kernel.cu"
#include "globals.h"
#include "consts.h"

#include "cuda_semillg.h"

#include <curand.h>
#include <cuda.h>
#include <cublas.h>
#include <cusparse.h>

#include <cmath>


#ifndef NDEBUG
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#else
#define CUDA_CALL(x) x
#endif

#ifndef NDEBUG
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
  exit(EXIT_FAILURE);}} while(0)
#else
#define CURAND_CALL(x) x
#endif

#if defined(__CUDACC__) && defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
    #error "-arch sm_13 nvcc flag is required to compile"
#endif

// block size for GPU, 64 appears to be most efficient for current kernel
#define BLOCKSIZE 32

void CUDASemiLLGSolver::initialise(int argc, char **argv, double idt)
{
  using namespace globals;

  // initialise base class
  Solver::initialise(argc,argv,idt);

  sigma.resize(nspins);

  for(int i=0; i<nspins; ++i) {
    sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)) / (dt*mus(i)) );
  }


  output.write("Initialising CUDA Semi Implicit LLG solver (CPU)\n");

  output.write("Initialising CUBLAS\n");

  output.write("Allocating device memory...\n");

  //-------------------------------------------------------------------
  //  Allocate device memory
  //-------------------------------------------------------------------

  // spin arrays
  CUDA_CALL(cudaMalloc((void**)&s_dev,nspins3*sizeof(s_dev)));
  CUDA_CALL(cudaMalloc((void**)&s_new_dev,nspins3*sizeof(s_new_dev)));

  // field arrays
  CUDA_CALL(cudaMalloc((void**)&h_dev,nspins3*sizeof(h_dev)));

  // wiener processes
  CUDA_CALL(cudaMalloc((void**)&w_dev,nspins3*sizeof(w_dev)));

  // jij matrix
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_row,(Jij.rows()+1)*sizeof(Jij_dev_row)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_col,Jij.nonzero()*sizeof(Jij_dev_col)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_val,Jij.nonzero()*sizeof(Jij_dev_val)));

  // material properties
  CUDA_CALL(cudaMalloc((void**)&mat_dev,4*nspins*sizeof(mat_dev)));
  //CUDA_CALL(cudaMalloc((void**)&mus_dev,nspins*sizeof(mus_dev)));
  //CUDA_CALL(cudaMalloc((void**)&gyro_dev,nspins*sizeof(gyro_dev)));
  //CUDA_CALL(cudaMalloc((void**)&alpha_dev,nspins*sizeof(alpha_dev)));
  //CUDA_CALL(cudaMalloc((void**)&sigma_dev,nspins*sizeof(sigma_dev)));

  //-------------------------------------------------------------------
  //  Copy data to device
  //-------------------------------------------------------------------

  output.write("Copying data to device memory...\n");
  // initial spins
  CUDA_CALL(cudaMemcpy(s_dev,s.ptr(),(size_t)(nspins3*sizeof(s_dev)),cudaMemcpyHostToDevice));

  // jij matrix
  CUDA_CALL(cudaMemcpy(Jij_dev_row,Jij.ptrRow(),
        (size_t)((Jij.rows()+1)*(sizeof(Jij_dev_row[0]))),cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(Jij_dev_col,Jij.ptrCol(),
        (size_t)((Jij.nonzero())*(sizeof(Jij_dev_col[0]))),cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(Jij_dev_val,Jij.ptrVal(),
        (size_t)((Jij.nonzero())*(sizeof(Jij_dev_val[0]))),cudaMemcpyHostToDevice));

  Array2D<double> mat(nspins,4);
  // material properties
  for(int i=0; i<nspins; ++i){
    mat(i,0) = mus(i);
    mat(i,1) = gyro(i);
    mat(i,2) = alpha(i);
    mat(i,3) = sigma(i);
  }
  CUDA_CALL(cudaMemcpy(mat_dev,mat.ptr(),(size_t)(4*nspins*sizeof(mat_dev)),cudaMemcpyHostToDevice));
  //CUDA_CALL(cudaMemcpy(mus_dev,mus.ptr(),(size_t)(nspins*sizeof(mus_dev)),cudaMemcpyHostToDevice));
  //CUDA_CALL(cudaMemcpy(gyro_dev,gyro.ptr(),(size_t)(nspins*sizeof(gyro_dev)),cudaMemcpyHostToDevice));
  //CUDA_CALL(cudaMemcpy(alpha_dev,alpha.ptr(),(size_t)(nspins*sizeof(alpha_dev)),cudaMemcpyHostToDevice));
  //CUDA_CALL(cudaMemcpy(sigma_dev,sigma.ptr(),(size_t)(nspins*sizeof(sigma_dev)),cudaMemcpyHostToDevice));

  //-------------------------------------------------------------------
  //  Initialise curand
  //-------------------------------------------------------------------

  // curand generator
  CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  //-------------------------------------------------------------------
  //  Initialise cusparse
  //-------------------------------------------------------------------
  cusparseStatus_t status;
  status = cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Library initialization failed");
  }

  // create matrix descriptor
  status = cusparseCreateMatDescr(&descra);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Matrix descriptor initialization failed");
  }
  cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);

  nblocks = (nspins+BLOCKSIZE-1)/BLOCKSIZE;

  initialised = true;
}

void CUDASemiLLGSolver::run()
{
  using namespace globals;

  // copy s_dev to s_new_dev
  // NOTE: this is part of the SEMILLG scheme
  CUDA_CALL(cudaMemcpy(s_new_dev,s_dev,(size_t)(nspins3*sizeof(s_dev)),cudaMemcpyDeviceToDevice));

  // generate wiener trajectories
  float stmp = sqrt(temperature);
  CURAND_CALL(curandGenerateNormal(gen, w_dev, nspins3, 0.0f, stmp));

  // calculate interaction fields (and zero field array)
  cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,nspins3,nspins3,1.0,descra,
      Jij_dev_val,Jij_dev_row,Jij_dev_col,s_dev,0.0,h_dev);

  // integrate
  cuda_semi_llg_kernelA<<<nblocks,BLOCKSIZE>>>
    (
      s_dev,
      h_dev,
      w_dev,
      mat_dev,
//      gyro_dev,
//      alpha_dev,
      Jij_dev_row,
      Jij_dev_col,
      Jij_dev_val,
      h_app[0],
      h_app[1],
      h_app[2],
      nspins,
      dt
    );

  // calculate interaction fields (and zero field array)
  cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,nspins3,nspins3,1.0,descra,
      Jij_dev_val,Jij_dev_row,Jij_dev_col,s_dev,0.0,h_dev);

  cuda_semi_llg_kernelB<<<nblocks,BLOCKSIZE>>>
    (
      s_dev,
      s_new_dev,
      h_dev,
      w_dev,
      mat_dev,
//      gyro_dev,
//      alpha_dev,
      Jij_dev_row,
      Jij_dev_col,
      Jij_dev_val,
      h_app[0],
      h_app[1],
      h_app[2],
      nspins,
      dt
    );

  if(iteration%1000 == 0){
  CUDA_CALL(cudaMemcpy(s.ptr(),s_dev,(size_t)(nspins3*sizeof(s_dev)),cudaMemcpyDeviceToHost));
  }

//  for(int i=0; i<nspins; ++i) {
//    std::cout<<i<<"\t"<<h(i,0)<<"\t"<<h(i,1)<<"\t"<<h(i,2)<<std::endl;
//  }

  iteration++;
}

CUDASemiLLGSolver::~CUDASemiLLGSolver()
{
  curandDestroyGenerator(gen);

  //-------------------------------------------------------------------
  //  Free device memory
  //-------------------------------------------------------------------

  // spin arrays
  CUDA_CALL(cudaFree(s_dev));
  CUDA_CALL(cudaFree(s_new_dev));

  // field arrays
  CUDA_CALL(cudaFree(h_dev));

  // wiener processes
  CUDA_CALL(cudaFree(w_dev));

  // jij matrix
  CUDA_CALL(cudaFree(Jij_dev_row));
  CUDA_CALL(cudaFree(Jij_dev_col));
  CUDA_CALL(cudaFree(Jij_dev_val));

  // material arrays
  //CUDA_CALL(cudaFree(mus_dev));
  //CUDA_CALL(cudaFree(gyro_dev));
  //CUDA_CALL(cudaFree(alpha_dev));
  CUDA_CALL(cudaFree(mat_dev));
}

