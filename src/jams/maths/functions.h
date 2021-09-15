// functions.h                                                         -*-C++-*-
#ifndef INCLUDED_JAMS_MATHS_FUNCTIONS
#define INCLUDED_JAMS_MATHS_FUNCTIONS

#include <cmath>

namespace jams {
namespace maths {
  double coth(double x);

  double gaussian(double x, double x0, double sigma, double A);
}
}

// ============================================================================
//                      INLINE FUNCTION DEFINITIONS
// ============================================================================

inline
double jams::maths::coth(double x) {
  return 1.0 / std::tanh(x);
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------