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
#define BLOCKSIZE 64

void CUDASemiLLGSolver::syncOutput()
{
  using namespace globals;
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaMemcpy(s.ptr(),s_dev,(size_t)(nspins3*sizeof(double)),cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaThreadSynchronize());
}

void CUDASemiLLGSolver::initialise(int argc, char **argv, double idt)
{
  using namespace globals;

  // initialise base class
  Solver::initialise(argc,argv,idt);

  sigma.resize(nspins);

  for(int i=0; i<nspins; ++i) {
    sigma(i) = sqrt( (2.0*boltzmann_si*alpha(i)) / (dt*mus(i)*mu_bohr_si) );
  }


  output.write("Initialising CUDA Semi Implicit LLG solver (CPU)\n");

  output.write("Initialising CUBLAS\n");

  output.write("Allocating device memory...\n");

  //-------------------------------------------------------------------
  //  Allocate device memory
  //-------------------------------------------------------------------

  // spin arrays
  CUDA_CALL(cudaMalloc((void**)&s_dev,nspins3*sizeof(double)));
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaMalloc((void**)&sf_dev,nspins3*sizeof(float)));
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaMalloc((void**)&s_new_dev,nspins3*sizeof(double)));
  CUDA_CALL(cudaThreadSynchronize());

  // field arrays
  CUDA_CALL(cudaMalloc((void**)&h_dev,nspins3*sizeof(float)));
  CUDA_CALL(cudaThreadSynchronize());

  if(nspins3%2 == 0) {
    // wiener processes
    CUDA_CALL(cudaMalloc((void**)&w_dev,nspins3*sizeof(float)));
  CUDA_CALL(cudaThreadSynchronize());
  } else {
    CUDA_CALL(cudaMalloc((void**)&w_dev,(nspins3+1)*sizeof(float)));
  CUDA_CALL(cudaThreadSynchronize());
  }


  // jij matrix
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_row,(Jij.rows()+1)*sizeof(int)));
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_col,Jij.nonZero()*sizeof(int)));
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_val,Jij.nonZero()*sizeof(float)));
  CUDA_CALL(cudaThreadSynchronize());

  // material properties
  CUDA_CALL(cudaMalloc((void**)&mat_dev,nspins*4*sizeof(float)));
  CUDA_CALL(cudaThreadSynchronize());

  //-------------------------------------------------------------------
  //  Copy data to device
  //-------------------------------------------------------------------

  output.write("Copying data to device memory...\n");
  // initial spins
  Array2D<float> sf(nspins,3);
  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sf(i,j) = static_cast<float>(s(i,j));
    }
  }
  CUDA_CALL(cudaMemcpy(s_dev,s.ptr(),(size_t)(nspins3*sizeof(double)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaMemcpy(sf_dev,sf.ptr(),(size_t)(nspins3*sizeof(float)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaThreadSynchronize());

  // jij matrix
  CUDA_CALL(cudaMemcpy(Jij_dev_row,Jij.rowPtr(),
        (size_t)((Jij.rows()+1)*(sizeof(int))),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaThreadSynchronize());

  CUDA_CALL(cudaMemcpy(Jij_dev_col,Jij.colPtr(),
        (size_t)((Jij.nonZero())*(sizeof(int))),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaThreadSynchronize());

  CUDA_CALL(cudaMemcpy(Jij_dev_val,Jij.valPtr(),
        (size_t)((Jij.nonZero())*(sizeof(float))),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaThreadSynchronize());

  Array2D<float> mat(nspins,4);
  // material properties
  for(int i=0; i<nspins; ++i){
    mat(i,0) = mus(i);
    mat(i,1) = gyro(i);
    mat(i,2) = alpha(i);
    mat(i,3) = sigma(i);
  }
  CUDA_CALL(cudaMemcpy(mat_dev,mat.ptr(),(size_t)(nspins*4*sizeof(float)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaThreadSynchronize());


  //-------------------------------------------------------------------
  //  Initialise arrays to zero
  //-------------------------------------------------------------------
  for(int i=0; i<nspins; ++i) {
    for(int j=0; j<3; ++j) {
      sf(i,j) = 0.0;
    }
  }
  
  CUDA_CALL(cudaMemcpy(w_dev,sf.ptr(),(size_t)(nspins3*sizeof(float)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaMemcpy(h_dev,sf.ptr(),(size_t)(nspins3*sizeof(float)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaThreadSynchronize());
  
  //-------------------------------------------------------------------
  //  Initialise curand
  //-------------------------------------------------------------------

  // curand generator
  //CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));


  // TODO: set random seed from config
  const unsigned long long gpuseed = rng.uniform()*18446744073709551615ULL;
  //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, gpuseed));
  CUDA_CALL(cudaThreadSynchronize());
  //CUDA_CALL(cudaThreadSetLimit(cudaLimitStackSize,1024));
  //CUDA_CALL(cudaThreadSynchronize());

  //-------------------------------------------------------------------
  //  Initialise cusparse
  //-------------------------------------------------------------------
  cusparseStatus_t status;
  status = cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Library initialization failed");
  }
  CUDA_CALL(cudaThreadSynchronize());

  // create matrix descriptor
  status = cusparseCreateMatDescr(&descra);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Matrix descriptor initialization failed");
  }
  CUDA_CALL(cudaThreadSynchronize());

  status = cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Matrix descriptor set type failed");
  }
  CUDA_CALL(cudaThreadSynchronize());
  status = cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Matrix descriptor set index base failed");
  }
  CUDA_CALL(cudaThreadSynchronize());
  /*
  status = cusparseSetMatFillMode(descra,CUSPARSE_FILL_MODE_UPPER);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Matrix descriptor set fill mode failed");
  }
  CUDA_CALL(cudaThreadSynchronize());
  status = cusparseSetMatDiagType(descra,CUSPARSE_DIAG_TYPE_NON_UNIT);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Matrix descriptor set diag type failed");
  }
  CUDA_CALL(cudaThreadSynchronize());
  */
  nblocks = (nspins+BLOCKSIZE-1)/BLOCKSIZE;

  initialised = true;
}

void CUDASemiLLGSolver::run()
{
  using namespace globals;

  // copy s_dev to s_new_dev
  // NOTE: this is part of the SEMILLG scheme
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaMemcpy(s_new_dev,s_dev,(size_t)(nspins3*sizeof(double)),cudaMemcpyDeviceToDevice));

  // generate wiener trajectories
  float stmp = sqrt(temperature);
  
  if(temperature > 0.0) {
    if(nspins3%2 == 0) {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, nspins3, 0.0f, stmp));
    } else {
      CURAND_CALL(curandGenerateNormal(gen, w_dev, (nspins3+1), 0.0f, stmp));
    }
  }
  CUDA_CALL(cudaThreadSynchronize());
  
  // calculate interaction fields (and zero field array)
  cusparseStatus_t stat =
  cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,nspins3,nspins3,1.0,descra,
      Jij_dev_val,Jij_dev_row,Jij_dev_col,sf_dev,0.0,h_dev);
  if(stat != CUSPARSE_STATUS_SUCCESS){
    jams_error("CUSPARSE FAILED\n");
}
  CUDA_CALL(cudaThreadSynchronize());

  // integrate
  cuda_semi_llg_kernelA<<<nblocks,BLOCKSIZE>>>
    (
      s_dev,
      sf_dev,
      h_dev,
      w_dev,
      mat_dev,
      h_app[0],
      h_app[1],
      h_app[2],
      nspins,
      dt
    );
  CUDA_CALL(cudaThreadSynchronize());

  // calculate interaction fields (and zero field array)
  cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,nspins3,nspins3,1.0,descra,
      Jij_dev_val,Jij_dev_row,Jij_dev_col,sf_dev,0.0,h_dev);
  CUDA_CALL(cudaThreadSynchronize());
  
  CUDA_CALL(cudaThreadSynchronize());
  cuda_semi_llg_kernelB<<<nblocks,BLOCKSIZE>>>
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
  CUDA_CALL(cudaThreadSynchronize());

  iteration++;
}

CUDASemiLLGSolver::~CUDASemiLLGSolver()
{
  CUDA_CALL(cudaThreadSynchronize());
  curandDestroyGenerator(gen);
  cusparseStatus_t status;

  status = cusparseDestroyMatDescr(descra);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE matrix destruction failed");
  }

  status = cusparseDestroy(handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    jams_error("CUSPARSE Library destruction failed");
  }

  //-------------------------------------------------------------------
  //  Free device memory
  //-------------------------------------------------------------------

  // spin arrays
  CUDA_CALL(cudaFree(s_dev));
  CUDA_CALL(cudaFree(sf_dev));
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
  CUDA_CALL(cudaFree(mat_dev));
}

