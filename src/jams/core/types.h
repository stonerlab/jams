// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_TYPES_H
#define JAMS_CORE_TYPES_H

#include "jams/containers/vec3.h"
#include "jams/containers/mat3.h"
#include "jams/helpers/utils.h"

//-----------------------------------------------------------------------------
// enums
//-----------------------------------------------------------------------------

enum class CoordinateFormat {CARTESIAN, FRACTIONAL};

inline CoordinateFormat coordinate_format_from_string(const std::string s) {
  if (capitalize(s) == "CART" || capitalize(s) == "CARTESIAN") return CoordinateFormat::CARTESIAN;
  if (capitalize(s) == "FRAC" || capitalize(s) == "FRACTIONAL") return CoordinateFormat::FRACTIONAL;
  throw std::runtime_error("Unknown coordinate format");
}

inline std::string to_string(CoordinateFormat f) {
  switch (f) {
    case CoordinateFormat::CARTESIAN:
      return "CARTESIAN";
    case CoordinateFormat::FRACTIONAL:
      return "FRACTIONAL";
  }
  throw std::invalid_argument("unknown CoordinateFormat");
}

enum OutputFormat {TEXT, HDF5};

//-----------------------------------------------------------------------------
// structs
//-----------------------------------------------------------------------------

struct Atom {
    int  id;
    int  material;
    int  motif_position;
    Vec3 fractional_pos;
};

template <typename T, typename Idx = int>
struct Triad {
    Idx i;
    Idx j;
    Idx k;
    T   value;
};

#endif  // JAMS_CORE_TYPES_H
