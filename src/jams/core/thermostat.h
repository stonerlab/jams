// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_THERMOSTAT_H
#define JAMS_CORE_THERMOSTAT_H

#include "jams/containers/multiarray.h"

#include <string>

class Thermostat {
 public:
  Thermostat(const double &temperature, const double &sigma, const int num_spins)
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
  static Thermostat* create(const std::string &thermostat_name);

  // accessors
  double temperature() const { return temperature_; }
  void set_temperature(const double T) { temperature_ = T; }

  virtual const double* device_data() { return noise_.device_data(); }
  virtual const double* data() { return noise_.data(); }

  virtual double field(int i, int j) { return noise_(i, j); }
 protected:
  double                  temperature_;
  jams::MultiArray<double, 2> sigma_;
  jams::MultiArray<double, 2> noise_;
};

#endif  // JAMS_CORE_THERMOSTAT_H
