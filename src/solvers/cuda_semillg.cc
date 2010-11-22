#include "globals.h"
#include "consts.h"

#include "cuda_semillg.h"

#include <curand.h>
#include <cuda.h>
#include <cublas.h>
#include <cusparse.h>

#include <cmath>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
  exit(EXIT_FAILURE);}} while(0)

void CUDASemiLLGSolver::initialise(int argc, char **argv, double idt)
{
  using namespace globals;

  // initialise base class
  Solver::initialise(argc,argv,idt);

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
  CUDA_CALL(cudaMalloc((void**)&h_new_dev,nspins3*sizeof(h_new_dev)));

  // wiener processes
  CUDA_CALL(cudaMalloc((void**)&w_dev,nspins3*sizeof(w_dev)));

  // jij matrix
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_row,(Jij.rows()+1)*sizeof(Jij_dev_row)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_col,Jij.nonzero()*sizeof(Jij_dev_col)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_val,Jij.nonzero()*sizeof(Jij_dev_val)));

  // material properties
  CUDA_CALL(cudaMalloc((void**)&mus_dev,nspins*sizeof(mus_dev)));
  CUDA_CALL(cudaMalloc((void**)&gyro_dev,nspins*sizeof(gyro_dev)));
  CUDA_CALL(cudaMalloc((void**)&alpha_dev,nspins*sizeof(alpha_dev)));

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

  // material properties
  CUDA_CALL(cudaMemcpy(mus_dev,mus.ptr(),(size_t)(nspins*sizeof(mus_dev)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(gyro_dev,mus.ptr(),(size_t)(nspins*sizeof(gyro_dev)),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(alpha_dev,mus.ptr(),(size_t)(nspins*sizeof(alpha_dev)),cudaMemcpyHostToDevice));

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

  initialised = true;
}

void CUDASemiLLGSolver::run()
{
  using namespace globals;

  // generate wiener trajectories
  float stmp = sqrt(temperature);
  CURAND_CALL(curandGenerateNormal(gen, w_dev, nspins3, 0.0f, 1.0f));

  cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,nspins3,nspins3,1.0,descra,
      Jij_dev_val,Jij_dev_row,Jij_dev_col,s_dev,0.0,h_dev);

  CUDA_CALL(cudaMemcpy(h.ptr(),h_dev,(size_t)(nspins3*sizeof(h_dev)),cudaMemcpyDeviceToHost));

  for(int i=0; i<nspins; ++i) {
    std::cout<<i<<"\t"<<h(i,0)<<"\t"<<h(i,1)<<"\t"<<h(i,2)<<std::endl;
  }

  iteration++;
}

CUDASemiLLGSolver::~CUDASemiLLGSolver()
{
  curandDestroyGenerator(gen);

  cudaFree(w_dev);
  //cublasFree(w_dev);
}

