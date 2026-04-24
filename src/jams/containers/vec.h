//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_VEC_H
#define JAMS_VEC_H

#include <array>
#include <complex>
#include <cstddef>

#include "jams/helpers/mixed_precision.h"

namespace jams {

template <typename T, std::size_t N>
using Vec = std::array<T, N>;

using Vec2 = Vec<double, 2>;

using Vec3 = Vec<double, 3>;
using Vec3f = Vec<float, 3>;
using Vec3R = Vec<Real, 3>;
using Vec3b = Vec<bool, 3>;
using Vec3i = Vec<int, 3>;
using Vec3cx = Vec<std::complex<double>, 3>;

using Vec4 = Vec<double, 4>;
using Vec4i = Vec<int, 4>;

} // namespace jams

template <typename T, std::size_t N>
using Vec = jams::Vec<T, N>;

using Vec2 = jams::Vec2;

using Vec3 = jams::Vec3;
using Vec3f = jams::Vec3f;
using Vec3R = jams::Vec3R;
using Vec3b = jams::Vec3b;
using Vec3i = jams::Vec3i;
using Vec3cx = jams::Vec3cx;

using Vec4 = jams::Vec4;
using Vec4i = jams::Vec4i;

#endif // JAMS_VEC_H
