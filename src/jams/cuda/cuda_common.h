//
// Created by Joseph Barker on 2019-02-07.
//

#ifndef JAMS_CUDA_ERROR_H
#define JAMS_CUDA_ERROR_H

#include <iostream>
#include <stdexcept>

#if HAS_CUDA

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <cufft.h>
#include <cassert>

#include "jams/helpers/error.h"

const char* cusparseGetStatusString(cusparseStatus_t status);
const char* curandGetStatusString(curandStatus_t status);
const char* cublasGetStatusString(cublasStatus_t status);
const char* cufftGetStatusString(cufftResult_t status);

#define CHECK_CUSPARSE_STATUS(x) \
{ \
  cusparseStatus_t stat; \
  if ((stat = (x)) != CUSPARSE_STATUS_SUCCESS) { \
    std::cerr << JAMS_FILE ": " << __PRETTY_FUNCTION__ << std::endl; \
    std::cerr << JAMS_ERROR_MESSAGE("cusparse returned ") << cusparseGetStatusString(stat) << std::endl; \
    jams_die("exiting"); \
  } \
}

#define CHECK_CURAND_STATUS(x) \
{ \
  curandStatus_t stat; \
  if ((stat = (x)) != CURAND_STATUS_SUCCESS) { \
    std::cerr << JAMS_FILE ": " << __PRETTY_FUNCTION__ << std::endl; \
    std::cerr << JAMS_ERROR_MESSAGE("curand returned ") << curandGetStatusString(stat) << std::endl; \
    jams_die("exiting"); \
  } \
}

#define CHECK_CUBLAS_STATUS(x) \
{ \
  cublasStatus_t stat; \
  if ((stat = (x)) != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << JAMS_FILE ": " << __PRETTY_FUNCTION__ << std::endl; \
    std::cerr << JAMS_ERROR_MESSAGE("cublas returned ") << cublasGetStatusString(stat) << std::endl; \
    jams_die("exiting"); \
  } \
}

#define CHECK_CUFFT_STATUS(x) \
{ \
  cufftResult_t stat; \
  if ((stat = (x)) != CUFFT_SUCCESS) { \
    std::cerr << JAMS_FILE ": " << __PRETTY_FUNCTION__ << std::endl; \
    std::cerr << JAMS_ERROR_MESSAGE("cufft returned ") << cufftGetStatusString(stat) << std::endl; \
    jams_die("exiting"); \
  } \
}


#define CHECK_CUDA_STATUS(x) \
{ \
  cudaError_t stat; \
  if ((stat = (x)) != cudaSuccess) { \
    std::cerr << JAMS_FILE ": " << __PRETTY_FUNCTION__ << std::endl; \
    std::cerr << JAMS_ERROR_MESSAGE("cuda returned ") << cudaGetErrorString(stat) << std::endl; \
    jams_die("exiting"); \
  } \
}

#ifdef NDEBUG
#define DEBUG_CHECK_CUDA_ASYNC_STATUS
#else
#define DEBUG_CHECK_CUDA_ASYNC_STATUS \
{ \
  CHECK_CUDA_STATUS(cudaPeekAtLastError());   \
  CHECK_CUDA_STATUS(cudaDeviceSynchronize()); \
}
#endif

template <typename T>
inline T* cuda_malloc_and_copy_to_device(const T* hst_ptr, unsigned array_size) {
  void* p;
  CHECK_CUDA_STATUS(cudaMalloc(&p, array_size*sizeof(T)));
  CHECK_CUDA_STATUS(cudaMemcpy(p, hst_ptr, array_size*sizeof(T), cudaMemcpyHostToDevice));
  return static_cast<T*>(p);
}

inline dim3 cuda_grid_size(const dim3 &block_size, const dim3 &grid_size) {
  return {(grid_size.x + block_size.x - 1) / block_size.x,
          (grid_size.y + block_size.y - 1) / block_size.y,
          (grid_size.z + block_size.z - 1) / block_size.z};
}


inline void cuda_throw(cudaError_t return_code, const char *file, int line) {
  using std::string;
  using std::to_string;
  if (return_code != cudaSuccess) {
    throw std::runtime_error(string(file) + "(" + to_string(line) + "): CUDA error: " + string(cudaGetErrorString(return_code)));
  }
}

#define IDX2D(i, j, size0, size1) \
  (i*size1+j)

#define IDX3D(i, j, k, size0, size1, size2) \
  ((i*size1+j)*size2+k)

#define IDX4D(i, j, k, l, size0, size1, size2, size3) \
  ((i*size1+j)*size2+k)*size3+l)

#endif // HAS_CUDA

#endif //JAMS_CUDA_ERROR_H
