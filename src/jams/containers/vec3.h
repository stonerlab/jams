//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_VEC3_H
#define JAMS_VEC3_H

#include <array>
#include <cmath>
#include <iosfwd>
#include <iomanip>
#include <algorithm>
#include <complex>
#include "jams/helpers/maths.h"

template <typename T, std::size_t N>
using Vec = std::array<T, N>;

using Vec3  = std::array<double, 3>;
using Vec3f = std::array<float, 3>;
using Vec3b = std::array<bool, 3>;
using Vec3i = std::array<int, 3>;
using Vec3cx  = std::array<std::complex<double>, 3>;

using Vec4  = std::array<double, 4>;
using Vec4i = std::array<int, 4>;

template <typename T>
inline constexpr Vec<T,3> operator-(const Vec<T,3>& rhs) {
  return {-rhs[0], -rhs[1], -rhs[2]};
}

template <typename T1, typename T2>
inline constexpr auto operator*(const T1& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs * rhs[0]), 3> {
  return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
}

template <typename T1, typename T2>
inline constexpr auto operator*(const Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] * rhs), 3> {
  return {lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs};
}

template <typename T1, typename T2>
inline constexpr auto operator/(const Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] / rhs), 3> {
  return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs};
}

template <typename T1, typename T2>
inline constexpr auto operator/(const T1& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs / rhs[0]), 3> {
  return {lhs / rhs[0], lhs / rhs[1], lhs / rhs[2]};
}

template <typename T1, typename T2>
inline constexpr auto operator+(const Vec<T1,3>& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs[0] + rhs[0]), 3> {
  return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

template <typename T1, typename T2>
inline constexpr auto operator-(const Vec<T1,3>& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs[0] - rhs[0]), 3> {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

template <typename T1, typename T2>
inline constexpr auto operator+=(Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] + rhs), 3> {
  return {lhs[0] += rhs, lhs[1] += rhs, lhs[2] += rhs};
}

template <typename T1, typename T2>
inline constexpr auto operator+=(Vec<T1,3>& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs[0] + rhs[0]), 3> {
  return {lhs[0] += rhs[0], lhs[1] += rhs[1], lhs[2] += rhs[2]};
}

template <typename T1, typename T2>
inline constexpr auto operator-=(Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] - rhs), 3> {
  return {lhs[0] -= rhs, lhs[1] -= rhs, lhs[2] -= rhs};
}

template <typename T1, typename T2>
inline constexpr auto operator-=(Vec<T1,3>& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs[0] - rhs[0]), 3> {
  return {lhs[0] -= rhs[0], lhs[1] -= rhs[1], lhs[2] -= rhs[2]};
}

template <typename T1, typename T2>
inline constexpr auto operator*=(Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] * rhs), 3> {
  return {lhs[0] *= rhs, lhs[1] *= rhs, lhs[2] *= rhs};
}

template <typename T1, typename T2>
inline constexpr auto operator/=(Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] / rhs), 3> {
  return {lhs[0] /= rhs, lhs[1] /= rhs, lhs[2] /= rhs};
}

template <typename T>
inline constexpr bool equal(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]);
}

template <typename T>
inline constexpr bool operator==(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return equal(lhs, rhs);
}

template <typename T>
inline constexpr bool operator!=(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return !equal(lhs, rhs);
}

template <typename T>
inline constexpr auto operator%(const Vec<T,3>& lhs, const Vec<T,3>& rhs) -> Vec<decltype(lhs[0] % rhs[0]), 3> {
  return {lhs[0] % rhs[0], lhs[1] % rhs[1], lhs[2] % rhs[2]};
}

template <typename T1, typename T2>
inline constexpr auto dot(const Vec<T1,3>& a, const Vec<T2,3>& b) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T1, typename T2>
inline constexpr auto dot(const Vec<T1,3>& a, const T2 b[3]) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T1, typename T2>
inline constexpr auto dot(const T1 a[3], const Vec<T2,3>& b) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T1, typename T2>
inline constexpr auto dot(const T1 a[3], const T2 b[3]) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T1>
inline constexpr auto abs(const Vec<T1,3>& a) -> Vec<decltype(std::abs(a[0])), 3> {
  return {std::abs(a[0]), std::abs(a[1]), std::abs(a[2])};
}

template <typename T1, typename T2>
inline constexpr auto cross(const Vec<T1,3>& a, const Vec<T2,3>& b) -> Vec<decltype(a[0] * b[0]), 3> {
  return {a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]};
}

template <typename T1, typename T2, typename T3>
inline constexpr auto scalar_triple_product(const Vec<T1,3>& a, const Vec<T2,3>& b, const Vec<T3,3>& c) -> decltype(a[0] * b[0] * c[0]) {
  return dot(a, cross(b, c));
}

template <typename T1, typename T2, typename T3>
inline constexpr auto vector_triple_product(const Vec<T1,3>& a, const Vec<T2,3>& b, const Vec<T3,3>& c) -> Vec<decltype(a[0] * b[0] * c[0]), 3> {
  return cross(a, cross(b, c));
}

template <typename T1, typename T2>
inline constexpr auto scale(const Vec<T1,3>& a, const Vec<T2,3>& b) -> Vec<decltype(a[0] * b[0]), 3> {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}



template <typename T>
inline constexpr auto norm(const Vec<T,3>& a) -> decltype(std::sqrt(a[0])) {
  return std::sqrt(dot(a, a));
}

template <typename T>
inline constexpr T norm_sq(const Vec<T,3>& a) {
  return dot(a, a);
}

template <typename T>
inline constexpr T angle(const Vec<T,3>& a, const Vec<T,3>& b) {
  return acos(dot(a,b) / (norm(a), norm(b)));
}

inline Vec3 spherical_to_cartesian_vector(const double r, const double theta, const double phi) {
  return {r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)};
}

inline Vec<double,3> cartesian_from_spherical(const double &r, const double &theta, const double &phi) {
  return {
      sin(theta) * cos(phi),
      sin(theta) * sin(phi),
      cos(theta)
  };
}

inline double azimuthal_angle(const Vec<double,3> a) {
  return acos(a[2]/norm(a));
}

inline double polar_angle(const Vec<double,3> a) {
  return atan2(a[2], a[0]);
}

template <typename T>
inline auto normalize(const Vec<T,3>& a) -> Vec<decltype(a[0] / std::abs(a[0])), 3> {
  return a / norm(a);
}

template <typename T>
inline bool approximately_zero(const Vec<T,3>& a, const T& epsilon = FLT_EPSILON) {
  for (auto n = 0; n < 3; ++n) {
    if (!approximately_zero(a[n], epsilon)) {
      return false;
    }
  }
  return true;
}

// Vec3 specialization
template <typename T>
inline bool approximately_equal(const Vec<T,3>& a, const Vec<T,3>& b, const T& epsilon = FLT_EPSILON) {
//  return approximately_equal(a[0], b[0], epsilon) && approximately_equal(a[1], b[1], epsilon) && approximately_equal(a[2], b[2], epsilon);
  for (auto n = 0; n < 3; ++n) {
    if (!approximately_equal(a[n], b[n], epsilon)) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline auto unit_vector(const Vec<T, 3> &a) -> Vec<decltype(a[0] / std::abs(a[0])), 3> {
  if (approximately_zero(a)) {
    return a;
  }

  return a / norm(a);
}

template <typename T>
inline constexpr T sum(const Vec<T,3>& a) {
  return a[0] + a[1] + a[2];
}

template <typename T>
inline constexpr T product(const Vec<T,3>& a) {
  return a[0] * a[1] * a[2];
}

template <typename T>
inline constexpr Vec<std::complex<T>,3> conj(const Vec<std::complex<T>,3>& a) {
  return {std::conj(a[0]), std::conj(a[1]), std::conj(a[2])};
}

template <typename T>
inline constexpr Vec<T,3> trunc(const Vec<T,3>& a) {
  return {std::trunc(a[0]), std::trunc(a[1]), std::trunc(a[2])};
}

template <typename T>
inline constexpr Vec<double,3> to_double(const Vec<T,3>& a) {
  return {
    static_cast<double>(std::trunc(a[0])),
    static_cast<double>(std::trunc(a[1])),
    static_cast<double>(std::trunc(a[2]))
  };
}

template <typename T>
inline constexpr Vec<int,3> to_int(const Vec<T,3>& a) {
  return {
      static_cast<int>(a[0]),
      static_cast<int>(a[1]),
      static_cast<int>(a[2])
  };
}

template <typename T, std::size_t N>
T abs_max(const std::array<T,N>& x) {
  return std::abs(*std::max_element(x.begin(), x.end(),
                           [](const T& a, const T& b) {
                               return std::abs(a) < std::abs(b); }));
}

template <typename T, std::size_t N>
Vec<T, N> normalize_components(const Vec<T, N>& x) {
  Vec<T, N> result;
  for (auto i = 0; i < N; ++i) {
    x[i] != 0 ? result[i] = x[i]/abs(x[i]) : result[i] = 0;
  }
  return result;
}


template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Vec<T,3> &a) {
  auto w = os.width();
  os << std::right << std::setw(w) << a[0] << " ";
  os << std::right << std::setw(w) << a[1] << " ";
  os << std::right << std::setw(w) << a[2];
  return os;
}


#endif //JAMS_VEC3_H
