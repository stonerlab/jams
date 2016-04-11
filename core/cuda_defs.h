// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDA_DEFS_H
#define JAMS_CORE_CUDA_DEFS_H

#include <exception>
#include <string>
#include <cstddef>
#include <map>

typedef double CudaFastFloat;

#define BLOCKSIZE 16
#define DIA_BLOCK_SIZE 256

typedef struct devDIA {
  int     *row;
  int     *col;
  CudaFastFloat   *val;
  size_t     pitch;
  int     blocks;
} devDIA;

typedef struct devCSR {
  int     *row;
  int     *col;
  CudaFastFloat   *val;
  int     blocks;
} devCSR;

typedef struct devCSRmap {
  int     *row;
  int     *col;
  int     *val;
  int     blocks;
} devCSRmap;

#ifdef DEBUG
#define cuda_api_error_check(ans) { cuda_throw((ans), __FILE__, __LINE__); }
#define cuda_kernel_error_check() { cuda_api_error_check(cudaPeekAtLastError()); cuda_api_error_check(cudaDeviceSynchronize()); }
#else
#define cuda_api_error_check(ans) { (ans); }
#define cuda_kernel_error_check() {  }
#endif

inline void cuda_throw(cudaError_t return_code, const char *file, int line) {
  using std::string;
  using std::to_string;
  if (return_code != cudaSuccess) {
    throw std::runtime_error(string(file) + "(" + to_string(line) + "): CUDA error: " + string(cudaGetErrorString(return_code)));
  }
}

#ifdef DEBUG
#define CUDA_CALL(x) do { if ((x) != cudaSuccess) { \
  printf("Error at %s:%d\n", __FILE__, __LINE__);\
    exit(EXIT_FAILURE); }} while (0)
#else
#define CUDA_CALL(x) x
#endif

#ifdef DEBUG
#define CURAND_CALL(x) do { if ((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n", __FILE__, __LINE__);\
  exit(EXIT_FAILURE); }} while (0)
#else
#define CURAND_CALL(x) x
#endif

#if defined(__CUDACC__) && defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
    #error "-arch sm_13 nvcc flag is required to compile"
#endif

#endif  // JAMS_CORE_CUDA_DEFS_H