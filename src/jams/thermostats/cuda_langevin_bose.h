// Copyright 2014 Joseph Barker. All rights reserved.

// This thermostat implementation is designed to reproduce a semiquantum thermostat
// which has a coth(omega) frequency dependence.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H

#if HAS_CUDA

#include <curand.h>
#include <fstream>
#include <mutex>

#include "jams/core/thermostat.h"

class CudaLangevinBoseThermostat : public Thermostat {
 public:
  CudaLangevinBoseThermostat(const double &temperature, const double &sigma, const int num_spins);
  ~CudaLangevinBoseThermostat();

  void update();

  // override the base class implementation
  const double* device_data() { return noise_.device_data(); }

 private:

    void warmup(const unsigned steps);

    bool debug_;
    bool do_zero_point_ = false;
    bool is_warmed_up_ = false;
    unsigned num_warm_up_steps_ = 0;

    jams::MultiArray<double, 1> zeta0_;
    jams::MultiArray<double, 1> zeta5_;
    jams::MultiArray<double, 1> zeta5p_;
    jams::MultiArray<double, 1> zeta6_;
    jams::MultiArray<double, 1> zeta6p_;
    jams::MultiArray<double, 1> eta0_;
    jams::MultiArray<double, 1> eta1a_;
    jams::MultiArray<double, 1> eta1b_;
    cudaStream_t                dev_stream_ = nullptr;
    cudaStream_t                dev_curand_stream_ = nullptr;
    double                      delta_tau_;
    double                      omega_max_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H
