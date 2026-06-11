// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/spatial_region.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <libconfig.h++>

#include "jams/helpers/maths.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

namespace {

constexpr double kRegionEpsilon = 1.0e-12;

struct ProjectionRange {
  double low = 0.0;
  double high = 0.0;
};

void require_valid_extent(const libconfig::Setting& setting,
                          const jams::Vec<double, 3>& size) {
  for (auto n = 0; n < 3; ++n) {
    if (!std::isfinite(size[n]) || size[n] <= 0.0) {
      throw jams::ConfigException(
          setting, "size", " must contain three finite positive lengths");
    }
  }
}

jams::Vec<double, 3> normalised_direction(const libconfig::Setting& setting) {
  auto direction = jams::read_vec_setting<double, 3>(setting["direction"], "direction");
  const double direction_norm = jams::norm(direction);
  if (!std::isfinite(direction_norm) || direction_norm <= 0.0) {
    throw jams::ConfigException(setting, "direction", " must be a finite non-zero vector");
  }

  return direction / direction_norm;
}

ProjectionRange projection_range_for_box(
    const jams::Vec<double, 3>& origin,
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

SpatialRegion::SpatialRegion(
    const SpatialRegionType type,
    jams::Vec<double, 3> origin,
    jams::Vec<double, 3> size,
    jams::Vec<double, 3> direction,
    const double projection_low,
    const double projection_high)
    : type_(type),
      origin_(origin),
      size_(size),
      direction_(direction),
      projection_low_(projection_low),
      projection_high_(projection_high) {}

SpatialRegion SpatialRegion::from_config(const libconfig::Setting& setting) {
  const auto type_name = lowercase(jams::config_required<std::string>(setting, "type"));
  const auto origin = jams::read_vec_setting<double, 3>(setting["origin"], "origin");
  const auto size = jams::read_vec_setting<double, 3>(setting["size"], "size");
  require_valid_extent(setting, size);

  if (type_name == "constant") {
    return SpatialRegion(
        SpatialRegionType::Constant, origin, size, {0.0, 0.0, 0.0}, 0.0, 0.0);
  }

  if (type_name != "linear") {
    throw jams::ConfigException(
        setting["type"], "type", " must be either 'constant' or 'linear'");
  }

  const auto direction = normalised_direction(setting);
  const auto projection_range = projection_range_for_box(origin, size, direction);
  if (projection_range.high <= projection_range.low) {
    throw jams::ConfigException(
        setting["direction"], "direction", " does not span the region extent");
  }

  return SpatialRegion(
      SpatialRegionType::Linear,
      origin,
      size,
      direction,
      projection_range.low,
      projection_range.high);
}

bool SpatialRegion::contains(const jams::Vec<double, 3>& position) const {
  for (auto n = 0; n < 3; ++n) {
    const double upper = origin_[n] + size_[n];
    if (position[n] < origin_[n] - kRegionEpsilon ||
        position[n] >= upper - kRegionEpsilon) {
      return false;
    }
  }
  return true;
}

double SpatialRegion::interpolation_fraction(const jams::Vec<double, 3>& position) const {
  if (!is_linear()) {
    return 0.0;
  }

  return clamped_fraction(
      jams::dot(position, direction_),
      projection_low_,
      projection_high_);
}

}  // namespace jams
