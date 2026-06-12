// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_THERMOSTAT_CLASSICAL_H
#define JAMS_THERMOSTAT_CLASSICAL_H

#include <random>
#include <vector>

#include <pcg_random.hpp>

#include "jams/core/thermostat.h"

class ThermostatClassical : public Thermostat {
 public:
  ThermostatClassical(const jams::Real& temperature,
                      const jams::Real& sigma,
                      jams::Real timestep,
                      int num_spins);

  void update() override;

 private:
  void ensure_random_generators(int count);

  jams::MultiArray<jams::Real, 1> sigma_spin_;
  pcg32_k1024 random_generator_;
  std::vector<pcg32_k1024> random_generators_;
};

#endif  // JAMS_THERMOSTAT_CLASSICAL_H
