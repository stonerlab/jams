// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_CLASSICAL_H
#define JAMS_CUDA_THERMOSTAT_CLASSICAL_H

#if HAS_CUDA

#include <curand.h>

#include "jams/core/thermostat.h"

class CudaThermostatClassical : public Thermostat {
 public:
  CudaThermostatClassical(const jams::Real &temperature, std::span<const double> coupling, const jams::Real timestep, const int num_spins);

  void update() override;
  bool supplies_white_field() const override { return true; }
  jams::ThermostatSamples samples() const override { return jams::ThermostatSamples::white_diffusion; }

  // Timestep-independent amplitude for integrators that draw increments.
  jams::MultiArray<double, 1> diffusion_amplitudes(double uniform_temperature) const override;

  cudaStream_t& get_stream() override;

  const jams::Real* device_data() override { return noise_.device_data(); }


  jams::Real field(int i, int j) {
    return noise_(i, j);
  }


 private:
    jams::MultiArray<jams::Real, 1> sigma_spin_;
    jams::MultiArray<jams::Real, 1> sigma_sqrt_temperature_;
    std::vector<double> coupling_;
    cudaStream_t                dev_stream_ = nullptr;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_CLASSICAL_H
