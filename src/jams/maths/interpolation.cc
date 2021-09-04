// interpolation.cc                                                    -*-C++-*-

#include "jams/maths/interpolation.h"

#include <cmath>
#include <cassert>

double
jams::maths::linear_interpolation(double x, double x1, double f_x1, double x2,
                                  double f_x2) {
  assert(x >= x1);
  assert(x <= x2);

#ifdef FP_FAST_FMA
  return std::fma((x - x1), (f_x2 - f_x1) / (x2 - x1), f_x2);
#else
  return f_x2 + (x - x1) * (f_x2 - f_x1) / (x2 - x1);
#endif
}


double
jams::maths::bilinear_interpolation(double x, double y, double x1, double y1,
                                    double x2,
                                    double y2, double f_11, double f_12,
                                    double f_21,
                                    double f_22) {

    double R1 = jams::maths::linear_interpolation(x, x1, f_11, x2, f_21);
    double R2 = jams::maths::linear_interpolation(x, x1, f_12, x2, f_22);

    return jams::maths::linear_interpolation(y, y1, R1, y2, R2);
}