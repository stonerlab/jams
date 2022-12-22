#ifndef JAMS_CUDA_CHECK_STATUS_H
#define JAMS_CUDA_CHECK_STATUS_H

#include <cuda_runtime.h>

#define CHECK_CUDA_STATUS(x) \
{ \
  cudaError_t stat; \
  if ((stat = (x)) != cudaSuccess) { \
    throw std::runtime_error(cudaGetErrorString(stat)); \
  } \
}

#endif //JAMS_CUDA_CHECK_STATUS_H
