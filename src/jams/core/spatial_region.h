// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SPATIAL_REGION_H
#define JAMS_CORE_SPATIAL_REGION_H

#include "jams/containers/vec3.h"
#include "jams/core/types.h"

namespace libconfig {
class Setting;
}

namespace jams {

enum class SpatialRegionType {
  Constant,
  Linear,
};

class SpatialRegion {
 public:
  static SpatialRegion from_config(const libconfig::Setting& setting);

  [[nodiscard]] SpatialRegionType type() const { return type_; }
  [[nodiscard]] bool is_constant() const { return type_ == SpatialRegionType::Constant; }
  [[nodiscard]] bool is_linear() const { return type_ == SpatialRegionType::Linear; }

  [[nodiscard]] bool contains(const jams::Vec<double, 3>& position) const;
  [[nodiscard]] double interpolation_fraction(const jams::Vec<double, 3>& position) const;

 private:
  SpatialRegion(
      SpatialRegionType type,
      CoordinateFormat coordinate_format,
      jams::Vec<double, 3> origin,
      jams::Vec<double, 3> size,
      jams::Vec<double, 3> direction,
      double projection_low,
      double projection_high);

  [[nodiscard]] jams::Vec<double, 3> region_position(
      const jams::Vec<double, 3>& cartesian_position) const;

  SpatialRegionType type_ = SpatialRegionType::Constant;
  CoordinateFormat coordinate_format_ = CoordinateFormat::FRACTIONAL;
  jams::Vec<double, 3> origin_{0.0, 0.0, 0.0};
  jams::Vec<double, 3> size_{0.0, 0.0, 0.0};
  jams::Vec<double, 3> direction_{0.0, 0.0, 0.0};
  double projection_low_ = 0.0;
  double projection_high_ = 0.0;
};

}  // namespace jams

#endif  // JAMS_CORE_SPATIAL_REGION_H
