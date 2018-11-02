// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDA_DEFS_H
#define JAMS_CORE_CUDA_DEFS_H

#include <exception>
#include <string>
#include <cstddef>
#include <map>

#include <cuda_runtime.h>

#define DIA_BLOCK_SIZE 256

inline dim3 cuda_grid_size(const dim3 &block_size, const dim3 &grid_size) {
  return {(grid_size.x + block_size.x - 1) / block_size.x,
          (grid_size.y + block_size.y - 1) / block_size.y,
          (grid_size.z + block_size.z - 1) / block_size.z};
}

typedef struct devDIA {
  int     *row;
  int     *col;
  double   *val;
  size_t     pitch;
  int     blocks;
} devDIA;

typedef struct devCSR {
  int     *row;
  int     *col;
  double   *val;
  int     blocks;
} devCSR;

#ifdef NDEBUG
#define cuda_api_error_check(ans) { (ans); }
#define cuda_kernel_error_check() {  }
#else
#define cuda_api_error_check(ans) { cuda_throw((ans), __FILE__, __LINE__); }
#define cuda_kernel_error_check() { cuda_api_error_check(cudaPeekAtLastError()); cuda_api_error_check(cudaDeviceSynchronize()); }
#endif

inline void cuda_throw(cudaError_t return_code, const char *file, int line) {
  using std::string;
  using std::to_string;
  if (return_code != cudaSuccess) {
    throw std::runtime_error(string(file) + "(" + to_string(line) + "): CUDA error: " + string(cudaGetErrorString(return_code)));
  }
}

#ifdef NDEBUG
#define CUDA_CALL(x) x
#else
#define CUDA_CALL(x) do { if ((x) != cudaSuccess) { \
  printf("Error at %s:%d\n", __FILE__, __LINE__);\
    exit(EXIT_FAILURE); }} while (0)
#endif

#ifdef NDEBUG
#define CURAND_CALL(x) x
#else
#define CURAND_CALL(x) do { if ((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n", __FILE__, __LINE__);\
  exit(EXIT_FAILURE); }} while (0)
#endif

#endif  // JAMS_CORE_CUDA_DEFS_H
