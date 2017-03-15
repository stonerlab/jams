// Copyright 2014 Joseph Barker. All rights reserved.

#include "jblib/math/summations.h"

float64 jblib::kahan_sum(const float64 * restrict data, const uint32 size) {
  // 'c' is a running compensation for lost low-order bits
  float64 sum = 0.0, c = 0.0;
  for (register uint32 i = 0; i != size; ++i) {
    register float64 y, t;
    y = data[i] - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}
