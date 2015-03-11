// Copyright 2014 Joseph Barker. All rights reserved.

// This thermostat implementation is designed to reproduce a semiquantum thermostat
// which has a coth(omega) frequency dependence.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_H

#ifdef CUDA

#include <curand.h>
#include <fstream>

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
    bool debug_;
    jblib::CudaArray<double, 1> dev_noise_;
    jblib::CudaArray<double, 1> dev_zeta_;
    jblib::CudaArray<double, 1> dev_eta_;
    curandGenerator_t           dev_rng_;  // device random generator
    cudaStream_t                dev_stream_;
    double                      tau_;
    double                      w_max_;
    std::ofstream               outfile_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_H
