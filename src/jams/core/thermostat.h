// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_THERMOSTAT_H
#define JAMS_CORE_THERMOSTAT_H

#include <memory>
#include <string>

#include "jams/containers/multiarray.h"
#include "jams/core/noise_generator.h"
#include "jams/helpers/mixed_precision.h"

#if HAS_CUDA
#include "jams/cuda/cuda_stream.h"
#endif

class Thermostat {
 public:
  Thermostat(std::unique_ptr<NoiseGenerator> generator,
             jams::MultiArray<jams::Real, 1> sigma);

  virtual ~Thermostat();

  void update();

  // factory
  static Thermostat* create(const std::string& thermostat_name, const jams::Real timestep);

  // accessors
  jams::Real temperature() const { return temperature_; }
  void set_temperature(const jams::Real T) { temperature_ = T; }

  const jams::Real* device_data() { return noise_.device_data(); }
  const jams::Real* data() { return noise_.data(); }

  jams::Real field(int i, int j) { return noise_(i, j); }

  NoiseGenerator* generator() const { return generator_.get(); }

#if HAS_CUDA
  cudaStream_t& get_stream() {
    return cuda_stream_.get();
  }

  // Call after enqueueing work to update the completion marker.
  void record_done() {
    cudaEventRecord(done_, cuda_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  // Make an external stream wait for this thermostat's work.
  void wait_on(cudaStream_t external) const {
    cudaStreamWaitEvent(external, done_, 0);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  // Host-side wait.
  void synchronize_done() const {
    cudaEventSynchronize(done_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
#endif

 private:
  jams::Real temperature_;
  std::unique_ptr<NoiseGenerator> generator_;
  jams::MultiArray<jams::Real, 1> sigma_;
  jams::MultiArray<jams::Real, 2> noise_;

#if HAS_CUDA
  CudaStream cuda_stream_{CudaStream::Priority::LOW};
  cudaEvent_t done_{};
#endif
};

#endif  // JAMS_CORE_THERMOSTAT_H
