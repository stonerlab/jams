// functions.cc                                                        -*-C++-*-

#include "jams/maths/functions.h"

#include <cmath>
#include <jams/helpers/maths.h>

///
/// The value at `x` of a gaussian with center `x0` width `sigma` and
/// amplitude `A`.
///
/// f(x) = A exp( (x-x0)^2 / ( 2 sigma^2))
///
double jams::maths::gaussian(double x, double x0, double sigma, double A) {
  double arg = (x - x0) / sigma;
  return A * exp(-0.5 * arg * arg);
}

double jams::maths::lorentzian(double x, double center, double width, double amplitude) {
  return (amplitude * width * x) / (pow2(pow2(center) - pow2(x)) + pow2(x * width));
}
