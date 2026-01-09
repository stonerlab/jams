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

class CudaThermostatQuantumSpde : public Thermostat {
 public:
  CudaThermostatQuantumSpde(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins);

  void update() override;

  // override the base class implementation
  const jams::Real* device_data() override { return noise_.device_data(); }

 private:
    // Generate random numbers on a low priority stream so it can be multiplexed with all of the field and integration
    // until we next need random numbers
    CudaStream curand_stream_{CudaStream::Priority::LOW};
    cudaEvent_t curand_done_{};
    bool debug_ = false;
    bool do_zero_point_ = false;

    jams::MultiArray<double, 1> zeta0_;
    jams::MultiArray<double, 1> zeta5_;
    jams::MultiArray<double, 1> zeta5p_;
    jams::MultiArray<double, 1> zeta6_;
    jams::MultiArray<double, 1> zeta6p_;
    jams::MultiArray<jams::Real, 1> eta0_;
    jams::MultiArray<jams::Real, 1> eta1a_;
    jams::MultiArray<jams::Real, 1> eta1b_;
    double                      delta_tau_;
    double                      omega_max_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H
