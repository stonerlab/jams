// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_THERMOSTAT_H
#define JAMS_CORE_THERMOSTAT_H

#include <string>

#include "jblib/containers/array.h"

class Thermostat {
 public:
  Thermostat(const double &temperature, const double &sigma, const int num_spins)
    : temperature_(temperature),
      sigma_(sigma),
      noise_(num_spins, 3)
  {}

  virtual ~Thermostat() {}
  virtual void update() = 0;

  // factory
  static Thermostat* create(const std::string &thermostat_name);

  // accessors
  double temperature() const { return temperature_; }
  void set_temperature(const double T) { temperature_ = T; }

  double sigma() const { return sigma_; }
  void set_sigma(const double S) { sigma_ = S;  }

  const double* noise() { return noise_.data(); }

 private:
  double                  temperature_;
  double                  sigma_;
  jblib::Array<double, 2> noise_;
};

#endif  // JAMS_CORE_THERMOSTAT_H
