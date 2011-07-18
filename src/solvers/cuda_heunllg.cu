
#include "cuda_spmv.h"
#include "cuda_heunllg_kernel.cu"
#include "globals.h"
#include "consts.h"

#include "cuda_heunllg.h"

#include <curand.h>
#include <cuda.h>
#include <cublas.h>
#include <cusparse.h>

#include <cmath>

// block size for GPU, 64 appears to be most efficient for current kernel
#define BLOCKSIZE 32

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

  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    jams_error("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
  }

  if(deviceCount == 0){
    jams_error("There is no device supporting CUDA\n");
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  
  output.write("  * CUDA Device compute capability %d.%d\n",deviceProp.major,deviceProp.minor);

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
  CUDA_CALL(cudaThreadSynchronize());
  CUDA_CALL(cudaThreadSetLimit(cudaLimitStackSize,1024));


  //-------------------------------------------------------------------
  //  Allocate device memory
  //-------------------------------------------------------------------

  output.write("  * Allocating device memory...\n");
  // spin arrays
  CUDA_CALL(cudaMalloc((void**)&s_dev,nspins3*sizeof(double)));
  CUDA_CALL(cudaMalloc((void**)&sf_dev,nspins3*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&s_new_dev,nspins3*sizeof(double)));

  // field arrays
  CUDA_CALL(cudaMalloc((void**)&h_dev,nspins3*sizeof(float)));

  if(nspins3%2 == 0) {
    // wiener processes
    CUDA_CALL(cudaMalloc((void**)&w_dev,nspins3*sizeof(float)));
  } else {
    CUDA_CALL(cudaMalloc((void**)&w_dev,(nspins3+1)*sizeof(float)));
  }


#ifdef FORCE_CUDA_DIA
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_row,(Jij.diags())*sizeof(int)));
//  CUDA_CALL(cudaMalloc((void**)&Jij_dev_val,(Jij.rows()*Jij.diags())*sizeof(float)));
  CUDA_CALL(cudaMallocPitch((void**)&Jij_dev_val,&diaPitch,(Jij.rows())*sizeof(float),Jij.diags()));
#else
  // jij matrix
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_row,(Jij.rows()+1)*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_col,Jij.nonZero()*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev_val,Jij.nonZero()*sizeof(float)));
#endif

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

#ifdef FORCE_CUDA_DIA
  CUDA_CALL(cudaMemcpy(Jij_dev_row,Jij.dia_offPtr(),
        (size_t)((Jij.diags())*(sizeof(int))),cudaMemcpyHostToDevice));
//  CUDA_CALL(cudaMemcpy(Jij_dev_val,Jij.valPtr(),
//        (size_t)((Jij.diags()*Jij.rows())*(sizeof(float))),cudaMemcpyHostToDevice));
//  diaPitch = Jij.rows();
   CUDA_CALL(cudaMemcpy2D(Jij_dev_val,diaPitch,Jij.valPtr(),Jij.rows()*sizeof(float),Jij.rows()*sizeof(float),Jij.diags(),cudaMemcpyHostToDevice));
   diaPitch = diaPitch/sizeof(float);

#else
  // jij matrix
  CUDA_CALL(cudaMemcpy(Jij_dev_row,Jij.rowPtr(),
        (size_t)((Jij.rows()+1)*(sizeof(int))),cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(Jij_dev_col,Jij.colPtr(),
        (size_t)((Jij.nonZero())*(sizeof(int))),cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(Jij_dev_val,Jij.valPtr(),
        (size_t)((Jij.nonZero())*(sizeof(float))),cudaMemcpyHostToDevice));
#endif

  Array2D<float> mat(nspins,4);
  // material properties
  for(int i=0; i<nspins; ++i){
    mat(i,0) = mus(i);
    mat(i,1) = gyro(i);
    mat(i,2) = alpha(i);
    mat(i,3) = sigma(i);
  }
  CUDA_CALL(cudaMemcpy(mat_dev,mat.ptr(),(size_t)(nspins*4*sizeof(float)),cudaMemcpyHostToDevice));


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

  //-------------------------------------------------------------------
  //  Initialise cusparse
  //-------------------------------------------------------------------

#ifndef FORCE_CUDA_DIA
  output.write("  * Initialising CUSPARSE...\n");
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
#endif

  nblocks = (nspins+BLOCKSIZE-1)/BLOCKSIZE;

  spmvblocks = std::min<int>(DIA_BLOCK_SIZE,(nspins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);

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
#ifdef FORCE_CUDA_DIA
  size_t offset = size_t(-1);
  CUDA_CALL(cudaBindTexture(&offset,tex_x_float,sf_dev));
//  if(offset !=0){
//    jams_error("Failed to bind texture");
//  }
  spmv_dia_kernel<<< spmvblocks, DIA_BLOCK_SIZE >>>(nspins3,nspins3,
    Jij.diags(),diaPitch,Jij_dev_row,Jij_dev_val,sf_dev,h_dev);
  CUDA_CALL(cudaUnbindTexture(tex_x_float));
  
#else
  cusparseStatus_t stat =
  cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,nspins3,nspins3,1.0,descra,
      Jij_dev_val,Jij_dev_row,Jij_dev_col,sf_dev,0.0,h_dev);
  if(stat != CUSPARSE_STATUS_SUCCESS){
    jams_error("CUSPARSE FAILED\n");
  }
#endif
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
#ifdef FORCE_CUDA_DIA
  CUDA_CALL(cudaBindTexture(&offset,tex_x_float,sf_dev));
//  if(offset !=0){
//    jams_error("Failed to bind texture");
//  }
  spmv_dia_kernel<<< spmvblocks, DIA_BLOCK_SIZE >>>(nspins3,nspins3,
    Jij.diags(),diaPitch,Jij_dev_row,Jij_dev_val,sf_dev,h_dev);
  CUDA_CALL(cudaUnbindTexture(tex_x_float));
#else
  cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,nspins3,nspins3,1.0,descra,
      Jij_dev_val,Jij_dev_row,Jij_dev_col,sf_dev,0.0,h_dev);
#endif
  
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

