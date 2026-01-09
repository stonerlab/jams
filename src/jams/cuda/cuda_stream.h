#ifndef JAMS_CUDA_WRAPPER_STREAM_H
#define JAMS_CUDA_WRAPPER_STREAM_H

#include <cassert>
#include <cuda_runtime.h>

#include "jams/cuda/cuda_common.h"

class CudaStream {
  public:
    enum class Priority
    {
      DEFAULT,
      HIGH,
      LOW
    };

    inline CudaStream();
    explicit inline CudaStream(std::nullptr_t);
    explicit inline CudaStream(Priority priority);

    inline ~CudaStream();

    inline CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
      other.stream_ = nullptr;
    }

    inline CudaStream& operator=(CudaStream&& other) noexcept {
      if (this != &other) {
        destroy_stream();
        stream_ = other.stream_;
        other.stream_ = nullptr;
      }
      return *this;
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    inline explicit operator bool() const;

    inline void synchronize();

    inline cudaStream_t& get();

  private:
    inline void create_stream();
    inline void destroy_stream();
    cudaStream_t stream_ = nullptr;
};

inline CudaStream::CudaStream() = default;

inline CudaStream::CudaStream(std::nullptr_t) {
}

inline CudaStream::CudaStream(Priority priority)
{
  int leastPriority = 0, greatestPriority = 0;
  cudaError_t result = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (result != cudaSuccess) {
    cuda_throw(result, __FILE__, __LINE__);
  }

  switch (priority)
  {
  case Priority::HIGH:
    result = cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, greatestPriority);
    break;
  case Priority::LOW:
    result = cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, leastPriority);
    break;
  case Priority::DEFAULT:
    result = cudaStreamCreate(&stream_);
    break;
  }

  if (result != cudaSuccess) {
    stream_ = nullptr;
    cuda_throw(result, __FILE__, __LINE__);
  }
}

inline CudaStream::~CudaStream() {
  destroy_stream();
}

inline void CudaStream::create_stream()
{
  assert(!stream_);
  cudaError_t result = cudaStreamCreate(&stream_);
  if (result != cudaSuccess) {
    stream_ = nullptr;
    cuda_throw(result, __FILE__, __LINE__);
  }
}

inline void CudaStream::destroy_stream()
{
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

inline cudaStream_t& CudaStream::get() {
  if (!stream_) {
    create_stream();
  }
  assert(stream_);
  return stream_;
}

inline void CudaStream::synchronize() {
  if (!stream_) {
    create_stream();
  }
  assert(stream_);
  CHECK_CUDA_STATUS(cudaStreamSynchronize(stream_));
}

inline CudaStream::operator bool() const {
  return stream_ != nullptr;
}

#endif // JAMS_CUDA_WRAPPER_STREAM_H
