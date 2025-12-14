// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_THERMOSTAT_H
#define JAMS_CORE_THERMOSTAT_H

#include "jams/containers/multiarray.h"

#include <string>

#include "jams/helpers/mixed_precision.h"


class Thermostat {
 public:
  Thermostat(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins)
    : temperature_(temperature),
      sigma_(num_spins, 3),
      noise_(num_spins, 3)
  {
    sigma_.zero();
    noise_.zero();
  }

  virtual ~Thermostat() {}
  virtual void update() = 0;

  // factory
  static Thermostat* create(const std::string &thermostat_name, const jams::Real timestep);

  // accessors
  jams::Real temperature() const { return temperature_; }
  void set_temperature(const jams::Real T) { temperature_ = T; }

  virtual const jams::Real* device_data() { return noise_.device_data(); }
  virtual const jams::Real* data() { return noise_.data(); }

  virtual jams::Real field(int i, int j) { return noise_(i, j); }
 protected:
  jams::Real                  temperature_;
  jams::MultiArray<jams::Real, 2> sigma_;
  jams::MultiArray<jams::Real, 2> noise_;
};

#endif  // JAMS_CORE_THERMOSTAT_H
