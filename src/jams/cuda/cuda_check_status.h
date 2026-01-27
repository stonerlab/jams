#ifndef JAMS_CUDA_CHECK_STATUS_H
#define JAMS_CUDA_CHECK_STATUS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CHECK_CUDA_STATUS(x) \
{ \
  cudaError_t stat = (x); \
  if (stat != cudaSuccess) { \
    throw std::runtime_error( \
      std::string(__FILE__) + "(" + std::to_string(__LINE__) + "): CUDA error: " + \
      std::string(cudaGetErrorString(stat))); \
  } \
}

#endif //JAMS_CUDA_CHECK_STATUS_H
