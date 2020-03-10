// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_RAND_H
#define JAMS_CORE_RAND_H

#include <random>
#include <array>
#include <pcg_random.hpp>
#include <sstream>
#include "jams/helpers/utils.h"

template <class RNG>
std::array<double, 3> uniform_random_sphere(RNG &gen) {
  std::uniform_real_distribution<> dist;
  double v1, v2, s;

  do {
    v1 = -1.0 + 2.0 * dist(gen);
    v2 = -1.0 + 2.0 * dist(gen);
    s = (v1 * v1) + (v2 * v2);
  } while (s > 1.0);

  auto ss = sqrt(1.0 - s);

  return {2.0 * v1 * ss, 2.0 * v2 * ss, 1.0 - 2.0 * s};
}

#endif // JAMS_CORE_RAND_H
