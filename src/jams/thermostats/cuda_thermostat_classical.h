// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_CLASSICAL_H
#define JAMS_CUDA_THERMOSTAT_CLASSICAL_H

#if HAS_CUDA

#include <curand.h>

#include "jams/core/thermostat.h"

class CudaThermostatClassical : public Thermostat {
 public:
  CudaThermostatClassical(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins);

  void update();

  const jams::Real* device_data() override { return noise_.device_data(); }


  jams::Real field(int i, int j) {
    return noise_(i, j);
  }


 private:
    cudaStream_t                dev_stream_ = nullptr;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_CLASSICAL_H
