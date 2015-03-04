// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_WHITE_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_WHITE_H

#ifdef CUDA

#include <curand.h>

#include "core/thermostat.h"

#include "jblib/containers/cuda_array.h"

class CudaLangevinWhiteThermostat : public Thermostat {
 public:
  CudaLangevinWhiteThermostat(const double &temperature, const double &sigma, const int num_spins);
  ~CudaLangevinWhiteThermostat();

  void update();

  // override the base class implementation
  const double* noise() { return dev_noise_.data(); }

 private:
    jblib::CudaArray<double, 1> dev_noise_;
    curandGenerator_t           dev_rng_;  // device random generator
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_WHITE_H
