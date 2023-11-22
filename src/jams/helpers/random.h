// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_RAND_H
#define JAMS_CORE_RAND_H

#include <random>
#include <array>
#include <pcg_random.hpp>
#include <sstream>
#include "jams/helpers/utils.h"

namespace jams {

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


template <class RNG>
inline std::array<std::array<double, 3>, 3> uniform_random_rotation(RNG &gen) {
  std::uniform_real_distribution<> dist;
  std::array<double, 3> v = uniform_random_sphere(gen);
  double theta = kTwoPi*dist(gen);
  double c = cos(theta);
  double s = sin(theta);

  return
      {(-1 + 2*v[0]*v[0])*c - 2*v[0]*v[1]*s, 2*v[0]*v[1]*c + (-1 + 2*v[0]*v[0])*s, 2*v[0]*v[2],
       2*v[0]*v[1]*c - (-1 + 2*v[1]*v[1])*s, (-1 + 2*v[1]*v[1])*c + 2*v[0]*v[1]*s, 2*v[1]*v[2],
       2*v[0]*v[2]*c - 2*v[1]*v[2]*s, 2*v[1]*v[2]*c + 2*v[0]*v[2]*s, -1 + 2*v[2]*v[2]};
}
}


#endif // JAMS_CORE_RAND_H
