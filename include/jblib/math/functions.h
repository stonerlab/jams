// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_MATH_FUNCTIONS_H
#define JBLIB_MATH_FUNCTIONS_H

#include "jblib/sys/define.h"
#include "jblib/sys/types.h"

namespace jblib {

  template <typename Type>
  inline Type square(const Type &x) {
    return (x*x);
  }

  inline float64 coth(const float64 &x) {
    return (1.0/tanh(x));
  }

  inline float64 brillouin_function(const float64 &x, const float64 &S){
    return (((2.0*S+1.0)/(2.0*S))*coth(((2.0*S+1.0)/(2.0*S))*x)
      - (1.0/(2.0*S))*coth((1.0/(2.0*S))*x));
  }

  inline float64 brillouin_function_derivative(const float64 &x, const float64 &S) {
    return square((1.0/(2.0*S))/sinh((1.0/(2.0*S))*x)) -square(((2.0*S+1.0)/(2.0*S))/sinh(((2.0*S+1.0)/(2.0*S))*x));
  }

  inline float64 langevin_function(const float64 &x) {
    return (coth(x) - (1.0/x));
  }
}  // namespace jblib

#endif  // JBLIB_MATH_FUNCTIONS_H
