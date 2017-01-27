// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_MATH_COORDINATES_H
#define JBLIB_MATH_COORDINATES_H

#include <cmath>

#include "jblib/sys/define.h"
#include "jblib/sys/types.h"

namespace jblib {

  inline void cartesian_to_spherical(const float64 &x, const float64 &y,
      const float64 &z, float64& r, float64& theta, float64& phi) {
    r       = sqrt(x*x+y*y+z*z);
    theta   = acos(z/r);
    phi     = atan2(y, x);
  }

  inline void spherical_to_cartesian(const float64 &r, const float64 &theta,
      const float64 &phi, float64& x, float64& y, float64& z) {
    x = r*cos(theta)*cos(phi);
    y = r*cos(theta)*sin(phi);
    z = r*sin(theta);
  }
}  // namespace jblib
#endif  // JBLIB_MATH_COORDINATES_H

