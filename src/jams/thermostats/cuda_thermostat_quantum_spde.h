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
  CudaThermostatQuantumSpde(const jams::Real &temperature, const jams::Real &sigma,
                           double timestep, int num_spins);

  void update() override;
  // Publish the t=0 sample before the first regular update. The caller
  // orders consumption with record_done() and wait_on() as for update().
  void sample_initial() override;

  // override the base class implementation
  const jams::Real* device_data() override { return noise_.device_data(); }

 private:
  bool initial_sample_ready_ = false;
  jams::MultiArray<jams::Real, 1> process_temperature_;
  std::unique_ptr<jams::CudaQuantumSpdeNoiseGenerator> noise_generator_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H
