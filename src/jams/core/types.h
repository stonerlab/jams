// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_TYPES_H
#define JAMS_CORE_TYPES_H

#include <complex>

#include "jams/containers/vec3.h"
#include "jams/containers/mat3.h"
#include "jams/helpers/utils.h"

using Complex = std::complex<double>;

//-----------------------------------------------------------------------------
// enums
//-----------------------------------------------------------------------------

enum class CoordinateFormat {CARTESIAN, FRACTIONAL};

//
// JAMS format:
// typename_i typename_j rx ry rz Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz
//
// KKR format:
// unitcell_pos_i unitcell_pos_j rx ry rz Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz
//
enum class InteractionFileFormat {UNDEFINED, JAMS, KKR};

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
    int  material_index;
    int  motif_index;
    Vec3 fractional_position;
};

template <typename T, typename Idx = int>
struct Triad {
    Idx i;
    Idx j;
    Idx k;
    T   value;
};

#endif  // JAMS_CORE_TYPES_H
