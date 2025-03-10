// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_RAND_H
#define JAMS_CORE_RAND_H

#include <random>
#include <array>
#include <pcg_random.hpp>
#include <sstream>
#include "jams/helpers/utils.h"

template <class RNG>
inline std::array<double, 3> uniform_random_sphere(RNG &gen) {
  std::uniform_real_distribution<> dist;
  double v1, v2, s;

  do {
    v1 = fma(2.0, dist(gen), -1.0);
    v2 = fma(2.0, dist(gen), -1.0);
    s = (v1 * v1) + (v2 * v2);
  } while (s > 1.0);

  auto ss = 2.0 * sqrt(1.0 - s);

  return {v1 * ss, v2 * ss, fma(-2.0, s, 1.0)};
}

#endif // JAMS_CORE_RAND_H
