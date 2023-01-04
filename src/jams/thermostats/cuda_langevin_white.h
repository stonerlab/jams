// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_WHITE_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_WHITE_H

#if HAS_CUDA

#include <curand.h>

#include "jams/core/thermostat.h"

class CudaLangevinWhiteThermostat : public Thermostat {
 public:
  CudaLangevinWhiteThermostat(const double &temperature, const double &sigma, const double timestep, const int num_spins);

  void update();

  const double* device_data() override { return noise_.device_data(); }


  double field(int i, int j) {
    return noise_(i, j);
  }


 private:
    cudaStream_t                dev_stream_ = nullptr;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_WHITE_H
