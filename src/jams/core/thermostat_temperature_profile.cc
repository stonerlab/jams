// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/thermostat_temperature_profile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <libconfig.h++>

#include "jams/containers/vec3.h"
#include "jams/core/globals.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

namespace {

constexpr double kRegionEpsilon = 1.0e-12;

void require_finite_nonnegative_temperature(const libconfig::Setting& setting,
                                            const char* name,
                                            const double temperature) {
  if (!std::isfinite(temperature) || temperature < 0.0) {
    throw jams::ConfigException(
        setting, name, " must be a finite non-negative temperature in Kelvin");
  }
}

void require_valid_extent(const libconfig::Setting& setting,
                          const jams::Vec<double, 3>& size) {
  for (auto n = 0; n < 3; ++n) {
    if (!std::isfinite(size[n]) || size[n] <= 0.0) {
      throw jams::ConfigException(
          setting, "size", " must contain three finite positive lengths");
    }
  }
}

bool contains_half_open(const jams::Vec<double, 3>& position,
                        const jams::Vec<double, 3>& origin,
                        const jams::Vec<double, 3>& size) {
  for (auto n = 0; n < 3; ++n) {
    const double upper = origin[n] + size[n];
    if (position[n] < origin[n] - kRegionEpsilon || position[n] >= upper - kRegionEpsilon) {
      return false;
    }
  }
  return true;
}

jams::Vec<double, 3> normalised_direction(const libconfig::Setting& setting) {
  auto direction = jams::read_vec_setting<double, 3>(setting["direction"], "direction");
  const double direction_norm = jams::norm(direction);
  if (!std::isfinite(direction_norm) || direction_norm <= 0.0) {
    throw jams::ConfigException(setting, "direction", " must be a finite non-zero vector");
  }

  return direction / direction_norm;
}

struct ProjectionRange {
  double low = 0.0;
  double high = 0.0;
};

ProjectionRange projection_range_for_box(const jams::Vec<double, 3>& origin,
                                         const jams::Vec<double, 3>& size,
                                         const jams::Vec<double, 3>& direction) {
  ProjectionRange range;
  for (auto n = 0; n < 3; ++n) {
    const double lower_face = origin[n];
    const double upper_face = origin[n] + size[n];
    if (direction[n] >= 0.0) {
      range.low += direction[n] * lower_face;
      range.high += direction[n] * upper_face;
    } else {
      range.low += direction[n] * upper_face;
      range.high += direction[n] * lower_face;
    }
  }
  return range;
}

double clamped_fraction(const double value, const double low, const double high) {
  if (high <= low) {
    return 0.0;
  }
  return std::clamp((value - low) / (high - low), 0.0, 1.0);
}

}  // namespace

namespace jams {

ThermostatTemperatureProfile::ThermostatTemperatureProfile(
    const int num_spins,
    const jams::Real uniform_temperature)
    : num_spins_(num_spins),
      uniform_temperature_(uniform_temperature) {
  if (!std::isfinite(static_cast<double>(uniform_temperature)) || uniform_temperature < 0.0) {
    throw std::runtime_error("thermostat temperature must be finite and non-negative");
  }
}

ThermostatTemperatureProfile ThermostatTemperatureProfile::from_config(
    const libconfig::Config& config,
    const int num_spins,
    const jams::Real fallback_temperature) {
  ThermostatTemperatureProfile profile(num_spins, fallback_temperature);

  if (!config.exists("thermostat")) {
    return profile;
  }

  const auto& thermostat_settings = config.lookup("thermostat");
  if (!thermostat_settings.exists("temperature_regions")) {
    return profile;
  }

  const auto& regions = thermostat_settings["temperature_regions"];
  if (!regions.isList() || regions.getLength() == 0) {
    throw jams::ConfigException(
        regions, "temperature_regions", " must be a non-empty list of region settings");
  }

  const bool has_default_temperature = thermostat_settings.exists("default_temperature");
  double default_temperature = static_cast<double>(fallback_temperature);
  if (has_default_temperature) {
    default_temperature = jams::config_required<double>(
        thermostat_settings, "default_temperature");
    require_finite_nonnegative_temperature(
        thermostat_settings["default_temperature"], "default_temperature", default_temperature);
  }

  std::vector<jams::Real> temperatures(
      num_spins, static_cast<jams::Real>(default_temperature));
  std::vector<int> region_assignment(num_spins, -1);

  for (auto region_index = 0; region_index < regions.getLength(); ++region_index) {
    const auto& region = regions[region_index];
    if (!region.isGroup()) {
      throw jams::ConfigException(region, "temperature region", " must be a group");
    }

    const auto type = lowercase(jams::config_required<std::string>(region, "type"));
    const auto origin = jams::read_vec_setting<double, 3>(region["origin"], "origin");
    const auto size = jams::read_vec_setting<double, 3>(region["size"], "size");
    require_valid_extent(region, size);

    if (type != "constant" && type != "linear") {
      throw jams::ConfigException(
          region["type"], "type", " must be either 'constant' or 'linear'");
    }

    double constant_temperature = 0.0;
    double low_temperature = 0.0;
    double high_temperature = 0.0;
    jams::Vec<double, 3> direction{0.0, 0.0, 0.0};
    ProjectionRange projection_range;

    if (type == "constant") {
      constant_temperature = jams::config_required<double>(region, "temperature");
      require_finite_nonnegative_temperature(
          region["temperature"], "temperature", constant_temperature);
    } else {
      low_temperature = jams::config_required<double>(region, "low");
      high_temperature = jams::config_required<double>(region, "high");
      require_finite_nonnegative_temperature(region["low"], "low", low_temperature);
      require_finite_nonnegative_temperature(region["high"], "high", high_temperature);
      direction = normalised_direction(region);
      projection_range = projection_range_for_box(origin, size, direction);
      if (projection_range.high <= projection_range.low) {
        throw jams::ConfigException(
            region["direction"], "direction", " does not span the region extent");
      }
    }

    int region_spin_count = 0;
    for (auto spin = 0; spin < num_spins; ++spin) {
      const jams::Vec<double, 3> position{
          static_cast<double>(globals::positions(spin, 0)),
          static_cast<double>(globals::positions(spin, 1)),
          static_cast<double>(globals::positions(spin, 2)),
      };

      if (!contains_half_open(position, origin, size)) {
        continue;
      }

      if (region_assignment[spin] >= 0) {
        std::ostringstream message;
        message << "temperature region " << region_index
                << " overlaps region " << region_assignment[spin]
                << " at spin " << spin;
        throw jams::ConfigException(region, message.str());
      }

      region_assignment[spin] = region_index;
      ++region_spin_count;

      if (type == "constant") {
        temperatures[spin] = static_cast<jams::Real>(constant_temperature);
      } else {
        const double projection = jams::dot(position, direction);
        const double u = clamped_fraction(
            projection, projection_range.low, projection_range.high);
        temperatures[spin] = static_cast<jams::Real>(
            low_temperature + u * (high_temperature - low_temperature));
      }
    }

    if (region_spin_count == 0) {
      std::ostringstream message;
      message << "temperature region " << region_index << " does not contain any spins";
      throw jams::ConfigException(region, message.str());
    }
  }

  if (!has_default_temperature) {
    for (auto spin = 0; spin < num_spins; ++spin) {
      if (region_assignment[spin] < 0) {
        std::ostringstream message;
        message << "temperature_regions do not cover spin " << spin
                << "; specify thermostat.default_temperature to allow uncovered spins";
        throw jams::ConfigException(regions, message.str());
      }
    }
  }

  profile.set_per_spin_temperatures(temperatures);
  return profile;
}

void ThermostatTemperatureProfile::set_uniform_temperature(const jams::Real temperature) {
  if (!std::isfinite(static_cast<double>(temperature)) || temperature < 0.0) {
    throw std::runtime_error("thermostat temperature must be finite and non-negative");
  }

  if (is_per_spin()) {
    throw std::runtime_error("cannot change a fixed per-spin thermostat temperature profile");
  }

  uniform_temperature_ = temperature;
}

void ThermostatTemperatureProfile::set_per_spin_temperatures(
    const std::vector<jams::Real>& temperatures) {
  mode_ = Mode::PerSpin;
  num_spins_ = static_cast<int>(temperatures.size());
  temperature_.resize(num_spins_);
  sqrt_temperature_.resize(num_spins_);

  for (auto spin = 0; spin < num_spins_; ++spin) {
    const auto temperature = temperatures[spin];
    if (!std::isfinite(static_cast<double>(temperature)) || temperature < 0.0) {
      throw std::runtime_error("thermostat temperature profile contains an invalid temperature");
    }

    temperature_(spin) = temperature;
    sqrt_temperature_(spin) = static_cast<jams::Real>(
        std::sqrt(static_cast<double>(temperature)));
  }
}

}  // namespace jams
