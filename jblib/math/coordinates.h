#ifndef JB_MATH_COORDINATES_H
#define JB_MATH_COORDINATES_H

#include <cmath>

#include "../sys/defines.h"
#include "../sys/types.h"

namespace jblib {

  JB_INLINE void cartesianToSpherical(const float64 x, const float64 y, 
      const float64 z, float64& r, float64& theta, float64& phi){
    r       = sqrt(x*x+y*y+z*z);
    theta   = acos( z/r );
    phi     = atan2(y,x);
  }

  JB_INLINE void sphericalToCartesian(const float64 r, const float64 theta,
      const float64 phi, float64& x, float64& y, float64& z){
    x = r*cos(theta)*cos(phi);
    y = r*cos(theta)*sin(phi);
    z = r*sin(theta);
  }

}
#endif

