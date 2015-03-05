// Copyright 2014 Joseph Barker. All rights reserved.

// This thermostat implementation is designed to reproduce a semiquantum thermostat
// which has a coth(omega) frequency dependence.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_H

#ifdef CUDA

#include <curand.h>

#include "core/thermostat.h"

#include "jblib/containers/cuda_array.h"

class CudaLangevinCothThermostat : public Thermostat {
 public:
  CudaLangevinCothThermostat(const double &temperature, const double &sigma, const int num_spins);
  ~CudaLangevinCothThermostat();

  void update();

  // override the base class implementation
  const double* noise() { return dev_noise_.data(); }

 private:
    jblib::CudaArray<double, 1> dev_noise_;
    jblib::CudaArray<double, 1> dev_zeta_linear_;
    jblib::CudaArray<double, 1> dev_zeta_bose_;
    jblib::CudaArray<double, 1> dev_eta_linear_;
    jblib::CudaArray<double, 1> dev_eta_bose_;
    curandGenerator_t           dev_rng_;  // device random generator
    cudaStream_t*               dev_streams_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_H
