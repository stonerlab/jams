#ifndef JAMS_CORE_NOISE_GENERATOR_H
#define JAMS_CORE_NOISE_GENERATOR_H

#include "jams/containers/multiarray.h"
#include "jams/helpers/mixed_precision.h"

#if HAS_CUDA
#include "jams/cuda/cuda_stream.h"
#endif

class NoiseGenerator {
 public:
  NoiseGenerator(const jams::Real& temperature, const int num_vectors)
      : temperature_(temperature),
        noise_(num_vectors, 3) {
    noise_.zero();

#if HAS_CUDA
    cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
#endif
  }

  virtual ~NoiseGenerator() {
#if HAS_CUDA
    if (done_ != nullptr) {
      cudaEventDestroy(done_);
      done_ = nullptr;
    }
#endif
  }

  virtual void update() = 0;

  jams::Real temperature() const { return temperature_; }
  void set_temperature(const jams::Real value) { temperature_ = value; }

  const jams::Real* device_data() { return noise_.device_data(); }
  const jams::Real* data() { return noise_.data(); }

  jams::Real field(int i, int j) { return noise_(i, j); }

  int num_vectors() const { return static_cast<int>(noise_.size(0)); }
  int num_channels() const { return static_cast<int>(noise_.elements()); }

#if HAS_CUDA
  cudaStream_t& get_stream() {
    return cuda_stream_.get();
  }

  void record_done() {
    cudaEventRecord(done_, cuda_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  void wait_on(cudaStream_t external) const {
    cudaStreamWaitEvent(external, done_, 0);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  void synchronize_done() const {
    cudaEventSynchronize(done_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
#endif

 protected:
  jams::Real temperature_;
  jams::MultiArray<jams::Real, 2> noise_;

#if HAS_CUDA
  CudaStream cuda_stream_{};
  cudaEvent_t done_{};
#endif
};

#endif  // JAMS_CORE_NOISE_GENERATOR_H
