#ifndef JAMS_LATTICE_COORDINATES_H
#define JAMS_LATTICE_COORDINATES_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "jams/containers/vec3.h"
#include "jams/helpers/defaults.h"

namespace jams::lattice {

/// Return @p value modulo @p period in the half-open integer range [0, period).
/// This is the integer-cell counterpart of wrapping fractional coordinates.
inline int modulo_index(const int value, const int period) {
  assert(period > 0);
  const int remainder = value % period;
  return remainder < 0 ? remainder + period : remainder;
}

/// Wrap one fractional coordinate into the canonical half-open unit-cell range
/// [0, 1). Values within @p tolerance of either cell face are snapped to zero so
/// round-off at a periodic boundary does not create a distinct coordinate.
inline double wrap_fractional_coordinate(
    const double value,
    const double tolerance = jams::defaults::lattice_tolerance) {
  if (!std::isfinite(value)) {
    return value;
  }

  double wrapped = value - std::floor(value);
  if (wrapped < 0.0) {
    wrapped += 1.0;
  }
  if (wrapped >= 1.0) {
    wrapped -= 1.0;
  }

  if (std::abs(wrapped) <= tolerance || std::abs(1.0 - wrapped) <= tolerance) {
    return 0.0;
  }
  return wrapped;
}

/// Wrap a fractional coordinate vector component-wise into [0, 1).
inline jams::Vec<double, 3> normalise_fractional_coordinate(
    jams::Vec<double, 3> r_frac,
    const double tolerance = jams::defaults::lattice_tolerance) {
  for (auto n = 0; n < 3; ++n) {
    r_frac[n] = wrap_fractional_coordinate(r_frac[n], tolerance);
  }
  return r_frac;
}

/// Return true when a fractional position is already in the canonical half-open
/// unit-cell range. Components near the upper face are treated as non-canonical
/// because they should be represented as zero in the next periodic image.
inline bool is_normalised_fractional_coordinate(
    const jams::Vec<double, 3>& r_frac,
    const double tolerance = jams::defaults::lattice_tolerance) {
  for (auto n = 0; n < 3; ++n) {
    if (!std::isfinite(r_frac[n])) {
      return false;
    }
    if (r_frac[n] < 0.0 || r_frac[n] >= 1.0 || std::abs(1.0 - r_frac[n]) <= tolerance) {
      return false;
    }
  }
  return true;
}

/// Compare free lattice vectors with an absolute component tolerance. This is
/// not modulo-periodic: integer translations remain distinct, which is required
/// for interaction vectors and symmetry-generated displacement vectors.
inline bool absolute_vector_equal(
    const jams::Vec<double, 3>& lhs,
    const jams::Vec<double, 3>& rhs,
    const double tolerance = jams::defaults::lattice_tolerance) {
  for (auto n = 0; n < 3; ++n) {
    if (std::abs(lhs[n] - rhs[n]) > tolerance) {
      return false;
    }
  }
  return true;
}

inline bool absolute_vector_exists_in_container(
    const std::vector<jams::Vec<double, 3>>& container,
    const jams::Vec<double, 3>& value,
    const double tolerance = jams::defaults::lattice_tolerance) {
  return std::any_of(container.begin(), container.end(), [&](const auto& existing) {
    return absolute_vector_equal(existing, value, tolerance);
  });
}

/// Compare fractional positions modulo the unit cell. Use this for positions,
/// not for displacement vectors where a whole-cell translation carries meaning.
inline bool fractional_positions_equivalent(
    const jams::Vec<double, 3>& lhs,
    const jams::Vec<double, 3>& rhs,
    const double tolerance = jams::defaults::lattice_tolerance) {
  return absolute_vector_equal(
      normalise_fractional_coordinate(lhs - rhs, tolerance),
      jams::Vec<double, 3>{0.0, 0.0, 0.0},
      tolerance);
}

/// Return the cell offset containing a fractional coordinate. Coordinates close
/// to an integer boundary are assigned to that boundary to avoid round-off
/// placing a point in the adjacent cell.
inline jams::Vec<double, 3> containing_cell_offset(
    const jams::Vec<double, 3>& r_frac,
    const double tolerance = jams::defaults::lattice_tolerance) {
  jams::Vec<double, 3> offset;
  for (auto n = 0; n < 3; ++n) {
    const double nearest_integer = std::nearbyint(r_frac[n]);
    if (std::abs(r_frac[n] - nearest_integer) <= tolerance) {
      offset[n] = nearest_integer;
    } else {
      offset[n] = std::floor(r_frac[n]);
    }
  }
  return offset;
}

/// Convert a fractional vector to an integer lattice vector when every
/// component is within @p tolerance of an integer and fits in the current index
/// type. Invalid interaction vectors should be rejected rather than truncated.
inline std::optional<jams::Vec<int, 3>> nearest_integer_lattice_vector(
    const jams::Vec<double, 3>& r_frac,
    const double tolerance = jams::defaults::lattice_tolerance) {
  jams::Vec<int, 3> result;
  for (auto n = 0; n < 3; ++n) {
    if (!std::isfinite(r_frac[n])) {
      return std::nullopt;
    }

    const double nearest_integer = std::nearbyint(r_frac[n]);
    if (std::abs(r_frac[n] - nearest_integer) > tolerance) {
      return std::nullopt;
    }
    if (nearest_integer < static_cast<double>(std::numeric_limits<int>::min())
        || nearest_integer > static_cast<double>(std::numeric_limits<int>::max())) {
      return std::nullopt;
    }

    result[n] = static_cast<int>(nearest_integer);
  }
  return result;
}

}  // namespace jams::lattice

#endif  // JAMS_LATTICE_COORDINATES_H
