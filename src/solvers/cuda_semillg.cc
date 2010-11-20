#include "globals.h"
#include "consts.h"

#include "cuda_semillg.h"

#include <curand.h>
#include <cuda.h>
#include <cublas.h>

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

  output.write("Allocating device memory\n");


  CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  
  CUDA_CALL(cudaMalloc((void**)&w_dev,nspins3*sizeof(float)));

  initialised = true;
}

void CUDASemiLLGSolver::run()
{
  using namespace globals;
  
  std::vector<float> tmp(nspins3);

  // generate wiener trajectories
  float stmp = 1.0;//sqrt(temperature);
  
  CURAND_CALL(curandGenerateUniform(gen, w_dev, nspins3));
  //CURAND_CALL(curandGenerateNormal(gen, w_dev, nspins3, 0.0f, 1.0f));
  
  CUDA_CALL(cudaMemcpy(&tmp[0],w_dev,nspins3*sizeof(float),cudaMemcpyDeviceToHost));
  for(int i=0; i<nspins3; ++i) {
    std::cout<<tmp[i]<<std::endl;
  }

  iteration++;
}

CUDASemiLLGSolver::~CUDASemiLLGSolver()
{
  curandDestroyGenerator(gen);

  cudaFree(w_dev);
  //cublasFree(w_dev);
}

