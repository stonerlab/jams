// Copyright 2014 Joseph Barker. All rights reserved.

// This thermostat implementation is designed to reproduce a semiquantum thermostat
// which has a coth(omega) frequency dependence.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H

#if HAS_CUDA

#include <memory>

#include "jams/core/thermostat.h"
#include "jams/thermostats/cuda_quantum_spde_noise.h"

class CudaThermostatQuantumSpde : public Thermostat {
 public:
  CudaThermostatQuantumSpde(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins);

  void update() override;

  // override the base class implementation
  const jams::Real* device_data() override { return noise_.device_data(); }

 private:
  std::unique_ptr<jams::CudaQuantumSpdeNoiseGenerator> noise_generator_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H
