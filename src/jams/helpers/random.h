// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_RAND_H
#define JAMS_CORE_RAND_H

#include <random>
#include <array>
#include <pcg_random.hpp>
#include <sstream>
#include "jams/helpers/utils.h"

template <typename T, class RNG>
inline std::array<T, 3> uniform_random_sphere(RNG &gen) {
  static_assert(std::is_arithmetic<T>::value,
                 "uniform_random_sphere requires arithmetic T type");

  std::uniform_real_distribution<T> dist;
  T v1, v2, s;

  do {
    v1 = fma(T(2.0), dist(gen), T(-1.0));
    v2 = fma(T(2.0), dist(gen), T(-1.0));
    s = (v1 * v1) + (v2 * v2);
  } while (s > T(1.0));

  auto ss = T(2.0) * sqrt(T(1.0) - s);

  return {v1 * ss, v2 * ss, fma(T(-2.0), s, T(1.0))};
}

#endif // JAMS_CORE_RAND_H
