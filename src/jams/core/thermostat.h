// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_THERMOSTAT_H
#define JAMS_CORE_THERMOSTAT_H

#include "jams/containers/multiarray.h"

#include <string>

#include "jams/helpers/mixed_precision.h"

#if HAS_CUDA
#include "jams/cuda/cuda_stream.h"
#endif


class Thermostat {
 public:
  Thermostat(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins)
    : temperature_(temperature),
      sigma_(num_spins, 3),
      noise_(num_spins, 3)
  {
    sigma_.zero();
    noise_.zero();

    cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  virtual ~Thermostat() = default;
  virtual void update() = 0;

  // factory
  static Thermostat* create(const std::string &thermostat_name, const jams::Real timestep);

  // accessors
  jams::Real temperature() const { return temperature_; }
  void set_temperature(const jams::Real T) { temperature_ = T; }

  virtual const jams::Real* device_data() { return noise_.device_data(); }
  virtual const jams::Real* data() { return noise_.data(); }

  virtual jams::Real field(int i, int j) { return noise_(i, j); }


#if HAS_CUDA
    cudaStream_t& get_stream()
  {
      return cuda_stream_.get();
  }

  // Call after enqueueing work to update the completion marker.
  void record_done()
  {
    cudaEventRecord(done_, cuda_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  // Make an external stream wait for this Hamiltonian's work.
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
 protected:
  jams::Real                  temperature_;
  jams::MultiArray<jams::Real, 2> sigma_;
  jams::MultiArray<jams::Real, 2> noise_;

#if HAS_CUDA
  CudaStream cuda_stream_ {};
  cudaEvent_t  done_{};
#endif
};

#endif  // JAMS_CORE_THERMOSTAT_H
