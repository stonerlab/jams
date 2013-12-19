#ifndef JB_CONSTANTS_H
#define JB_CONSTANTS_H

#include "../sys/types.h"

#define JB_RAND_MAX UINT32_MAX

namespace jblib{
  const float64 floatEpsilon = 1e-8;
  const float64 maxfloat64    = 1.7976931348623157e308;

  const float64 twopi  = 6.2831853071795864769252867663;
  const float64 pi     = 3.1415926535897932384626433832795;
  const float64 halfpi = 1.5707963267948965579989817342721;

  const float64 bohrMagneton = 9.27400915e-24;
  const float64 boltzmann    = 1.3806504e-23;
  const float64 hbar         = 1.05457148e-34;
}

#endif
