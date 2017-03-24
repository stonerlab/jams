#ifndef JAMS_CUDA_STREAM_FACADE_H
#define JAMS_CUDA_STREAM_FACADE_H

#include <cuda_runtime.h>

#include "jams/core/cuda_defs.h"

class CudaStreamFacade {
  public:
    CudaStreamFacade();
    ~CudaStreamFacade();

    bool is_allocated() const;

    void create();
    void destroy();
    void synchronize();

    cudaStream_t& get();

  private:
    cudaStream_t stream_ = nullptr;
};


inline CudaStreamFacade::CudaStreamFacade() {
}

inline CudaStreamFacade::~CudaStreamFacade() {
  synchronize();
  destroy();
}

inline cudaStream_t& CudaStreamFacade::get() {
  if (!is_allocated()) {
    create();
  }
  return stream_;
}

inline void CudaStreamFacade::create() {
  if (!is_allocated()) {
    cuda_api_error_check(cudaStreamCreate(&stream_));
  }
}

inline void CudaStreamFacade::destroy() {
  if (is_allocated()) {
    cuda_api_error_check(cudaStreamDestroy(stream_));
  }
}

inline void CudaStreamFacade::synchronize() {
  if(is_allocated()) {
    cuda_api_error_check(cudaStreamSynchronize(stream_));
  }
}

inline bool CudaStreamFacade::is_allocated() const {
  return stream_ != nullptr;
}

#endif // JAMS_CUDA_STREAM_FACADE_H
