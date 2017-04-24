// Copyright 2014 Joseph Barker. All rights reserved.

#include "crystal.h"

#include "jblib/sys/assert.h"
#include "jblib/sys/types.h"

#include "equalities.h"
#include "jblib/containers/vec3.h"

std::ostream& operator<<(std::ostream& os, enum crystal_t x) {
  switch (x){
    case cubic:
      return os << "cubic";
      break;
    case tetragonal:
      return os << "tetragonal";
      break;
    case orthorhombic:
      return os << "orthorhombic";
      break;
    case monoclinic:
      return os << "monoclinic";
      break;
    case triclinic:
      return os << "triclinic";
      break;
    case trigonal:
      return os << "trigonal";
      break;
    case hexagonal:
      return os << "hexagonal";
      break;
    case undefined:
      return os << "undefined";
      break;
    default:
      return os << "undefined";
  }
}

namespace jblib {
  crystal_t identify_crystal_system(const float64 v0[3], const float64 v1[3], const float64 v2[3]){

    float64 a, b, c;               // lengths of basis vectors
    float64 alpha, beta, gamma;    // angles between basis vectors

    a = jbVec::mod(v0);
    b = jbVec::mod(v1);
    c = jbVec::mod(v2);

    alpha = jbVec::angle(v0, v1);
    beta  = jbVec::angle(v1, v2);
    gamma = jbVec::angle(v0, v2);

    if (three_floats_are_equal(a, b, c)) {
      assert(three_floats_are_equal(alpha, beta, gamma));
      if (floats_are_equal(alpha, half_pi)) {
        return cubic;
      } else {
        return trigonal;
      }
    } else if (three_floats_are_different(a, b, c)) {
      if (three_floats_are_equal(alpha, beta, gamma)) {
        return orthorhombic;
      } else if (three_floats_are_different(alpha, beta, gamma)) {
        return triclinic;
      } else {
        return monoclinic;
      }
    } else {
      if (three_floats_are_equal(alpha, beta, gamma)) {
        return tetragonal;
      } else {
        return hexagonal;
      }
    }
    return undefined;
  }
}  // namespace jblib