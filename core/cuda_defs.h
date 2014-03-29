// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDA_DEFS_H
#define JAMS_CORE_CUDA_DEFS_H

#include <cstddef>

typedef double CudaFastFloat;

#define BLOCKSIZE 64
#define DIA_BLOCK_SIZE 256

typedef struct devDIA {
  int     *row;
  int     *col;
  CudaFastFloat   *val;
  size_t     pitch;
  int     blocks;
} devDIA;

typedef struct devCSR {
  int     *pointers;
  int     *coords;
  CudaFastFloat   *val;
  int     blocks;
} devCSR;


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