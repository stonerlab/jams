#ifndef JAMS_CUDA_WRAPPER_STREAM_H
#define JAMS_CUDA_WRAPPER_STREAM_H

#include <cassert>
#include <cuda_runtime.h>

#include "jams/cuda/cuda_defs.h"

class CudaStream {
  public:
    CudaStream();
    CudaStream(std::nullptr_t);

    ~CudaStream();

    CudaStream(CudaStream&&) = default;
    CudaStream& operator=(CudaStream&&) = default;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    explicit operator bool() const;

    void synchronize();

    cudaStream_t& get();

  private:
    cudaStream_t stream_ = nullptr;
};

inline CudaStream::CudaStream() {
  cudaError_t result = cudaStreamCreate(&stream_);
  if (result != cudaSuccess) {
    stream_ = nullptr;
    cuda_throw(result, __FILE__, __LINE__);
  }
}

inline CudaStream::CudaStream(std::nullptr_t) {
}

inline CudaStream::~CudaStream() {
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

inline cudaStream_t& CudaStream::get() {
  assert(stream_);
  return stream_;
}

inline void CudaStream::synchronize() {
  if(stream_) {
    cuda_api_error_check(cudaStreamSynchronize(stream_));
  }
}

inline CudaStream::operator bool() const {
  return stream_ != nullptr;
}

#endif // JAMS_CUDA_WRAPPER_STREAM_H
