#ifndef JAMS_SOLVERS_LLG_RKMK_THERMAL_KICK_H
#define JAMS_SOLVERS_LLG_RKMK_THERMAL_KICK_H

#include "jams/core/thermostat.h"
#include "jams/helpers/consts.h"
#include "jams/solvers/llg_rkmk_functions.h"

namespace jams::solvers::rkmk {

// Bind the correction to the same site temperatures that scaled the field
// draw. On a uniform bath this takes only one scalar per half-step.
inline ThermalKick make_thermal_kick(const Thermostat& bath,
                                     const MultiArray<Real, 1>& inverse_moments,
                                     double dt, bool automatic, bool device = false) {
  if (!automatic || !bath.supplies_white_field()) return {};

  const auto* inverse_moment_values = device ? inverse_moments.device_data() : inverse_moments.host_data();
  if (bath.has_uniform_temperature()) {
    return {inverse_moment_values, nullptr, kBoltzmannIU * dt * bath.temperature()};
  }

  const auto& temperatures = bath.temperature_profile().temperature();
  const auto* temperature_values = device ? temperatures.device_data() : temperatures.host_data();
  return {inverse_moment_values, temperature_values, kBoltzmannIU * dt};
}

}  // namespace jams::solvers::rkmk

#endif  // JAMS_SOLVERS_LLG_RKMK_THERMAL_KICK_H
