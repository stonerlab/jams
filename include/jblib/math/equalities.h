// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_MATH_EQUALITIES_H
#define JBLIB_MATH_EQUALITIES_H

#include <cmath>

#include "jblib/sys/define.h"
#include "jblib/sys/types.h"

#include "jblib/math/constants.h"

namespace jblib {

  inline bool floats_are_equal(const float64 &x, const float64 &y,
    const float64 &epsilon = float_epsilon) {
    return ( fabs( x - y ) < epsilon );
  }

  inline bool floats_are_less_than_or_equal(const float64 &x, const float64 &y,
    const float64 &epsilon = float_epsilon) {
    return ((x < y) || fabs( x - y ) < epsilon );
  }

  inline bool floats_are_greater_than_or_equal(const float64 &x,
    const float64 &y, const float64 &epsilon = float_epsilon) {
    return ((x > y) || fabs( x - y ) < epsilon );
  }


  inline bool three_floats_are_equal(const float64 &x, const float64 &y,
    const float64 &z, const float64 &epsilon = float_epsilon) {
    return (floats_are_equal(x, y, epsilon) && floats_are_equal(y, z, epsilon));
  }

  inline bool three_floats_are_different(const float64 &x, const float64 &y,
    const float64 &z, const float64 &epsilon = float_epsilon) {
    return ((!floats_are_equal(x, y, epsilon) && !floats_are_equal(y, z, epsilon)));
  }
}  // namespace jblib

#endif  // JBLIB_MATH_EQUALITIES_H
