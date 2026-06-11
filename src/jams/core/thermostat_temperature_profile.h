// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_THERMOSTAT_TEMPERATURE_PROFILE_H
#define JAMS_CORE_THERMOSTAT_TEMPERATURE_PROFILE_H

#include "jams/containers/multiarray.h"
#include "jams/helpers/mixed_precision.h"

#include <vector>

namespace libconfig {
class Config;
}

namespace jams {

class ThermostatTemperatureProfile {
 public:
  enum class Mode {
    Uniform,
    PerSpin,
  };

  ThermostatTemperatureProfile() = default;
  ThermostatTemperatureProfile(int num_spins, jams::Real uniform_temperature);

  static ThermostatTemperatureProfile from_config(
      const libconfig::Config& config,
      int num_spins,
      jams::Real fallback_temperature);

  [[nodiscard]] bool is_uniform() const { return mode_ == Mode::Uniform; }
  [[nodiscard]] bool is_per_spin() const { return mode_ == Mode::PerSpin; }
  [[nodiscard]] int size() const { return num_spins_; }

  [[nodiscard]] jams::Real uniform_temperature() const {
    return uniform_temperature_;
  }

  void set_uniform_temperature(jams::Real temperature);

  [[nodiscard]] const jams::MultiArray<jams::Real, 1>& temperature() const {
    return temperature_;
  }

  [[nodiscard]] const jams::MultiArray<jams::Real, 1>& sqrt_temperature() const {
    return sqrt_temperature_;
  }

 private:
  void set_per_spin_temperatures(const std::vector<jams::Real>& temperatures);

  Mode mode_ = Mode::Uniform;
  int num_spins_ = 0;
  jams::Real uniform_temperature_ = 0.0;
  jams::MultiArray<jams::Real, 1> temperature_;
  jams::MultiArray<jams::Real, 1> sqrt_temperature_;
};

}  // namespace jams

#endif  // JAMS_CORE_THERMOSTAT_TEMPERATURE_PROFILE_H
