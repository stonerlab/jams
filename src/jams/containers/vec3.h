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
#include <type_traits>
#include "jams/helpers/maths.h"
#include "jams/helpers/mixed_precision.h"

template <typename T, std::size_t N>
using Vec = std::array<T, N>;

using Vec2  = std::array<double, 2>;

using Vec3  = std::array<double, 3>;
using Vec3f = std::array<float, 3>;
using Vec3R = std::array<jams::Real, 3>;
using Vec3b = std::array<bool, 3>;
using Vec3i = std::array<int, 3>;
using Vec3cx  = std::array<std::complex<double>, 3>;

using Vec4  = std::array<double, 4>;
using Vec4i = std::array<int, 4>;



template <typename To, typename From, std::size_t N>
constexpr std::array<To, N>
array_cast(const std::array<From, N>& in)
{
  if constexpr (std::is_same<To, From>::value) {
    return in;
  } else {
    static_assert(std::is_arithmetic<To>::value,
                  "array_cast requires arithmetic To type");
    static_assert(std::is_arithmetic<From>::value,
                  "array_cast requires arithmetic From type");

    std::array<To, N> out{};
    for (std::size_t i = 0; i < N; ++i)
      out[i] = static_cast<To>(in[i]);
    return out;
  }
}

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


/// Returns true if all components of the Vec are exactly equal, false otherwise.
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


/// Returns the fused-multiply-add operation elementwise on the vectors a,b and c.
/// x_k = (a_k * b_k) + c_k
template <typename T1>
inline Vec<T1,3> fma(const Vec<T1,3>& a, const Vec<T1,3>& b, const Vec<T1,3>& c) {
  return {std::fma(a[0], b[0], c[0]), std::fma(a[1], b[1], c[1]), std::fma(a[2], b[2], c[2])};
}

/// Returns the fused-multiply-add operation elementwise on a,b and c where
/// a is a scalar, b and c are vectors.
/// x_k = (a * b_k) + c_k
template <typename T1>
inline Vec<T1,3> fma(const T1& a, const Vec<T1,3>& b, const Vec<T1,3>& c) {
  return {std::fma(a, b[0], c[0]), std::fma(a, b[1], c[1]), std::fma(a, b[2], c[2])};
}

/// Returns the dot product a . b
template <typename T1, typename T2>
inline constexpr auto dot(const Vec<T1,3>& a, const Vec<T2,3>& b) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Returns the dot product a . b
template <typename T1, typename T2>
inline constexpr auto dot(const Vec<T1,3>& a, const T2 b[3]) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Returns the dot product a . b
template <typename T1, typename T2>
inline constexpr auto dot(const T1 a[3], const Vec<T2,3>& b) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Returns the dot product a . b
template <typename T1, typename T2>
inline constexpr auto dot(const T1 a[3], const T2 b[3]) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Returns the dot product of a and b, which is then squared.
template <typename T1, typename T2>
inline constexpr auto dot_squared(const Vec<T1,3>& a, const Vec<T2,3>& b) -> decltype(a[0] * b[0]) {
  return pow2(dot(a,b));
}

/// Returns the Euclidean norm \sqrt(x^2 + y^2 + z^2) of the vector.
template <typename T>
inline constexpr auto norm(const Vec<T,3>& a) -> decltype(std::sqrt(a[0])) {
  return std::sqrt(dot(a, a));
}

/// Returns the square of the Euclidean norm (x^2 + y^2 + z^2) of the vector.
template <typename T>
inline constexpr T norm_squared(const Vec<T,3>& a) {
  return dot(a, a);
}

/// Returns a vector of the absolute values of each component of the argument
/// vector 'a'.
template <typename T1>
inline constexpr auto absolute(const Vec<T1,3>& a) -> Vec<decltype(std::abs(a[0])), 3> {
  return {std::abs(a[0]), std::abs(a[1]), std::abs(a[2])};
}

/// Returns a Vec from the cross product a x b.
template <typename T1, typename T2>
inline constexpr auto cross(const Vec<T1,3>& a, const Vec<T2,3>& b) -> Vec<decltype(a[0] * b[0]), 3> {
  return {a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]};
}


/// Returns the magnitude |a x b|^2 using the identity |a||b| - |a.b|^2
template <typename T1, typename T2>
inline constexpr auto cross_norm_squared(const Vec<T1,3>& a, const Vec<T2,3>& b) -> decltype(a[0] * b[0]) {
  return norm_squared(a) * norm_squared(b) - dot_squared(a, b);
}

/// Returns the scalar triple product a . (b x c)
template <typename T1, typename T2, typename T3>
inline constexpr auto scalar_triple_product(const Vec<T1,3>& a, const Vec<T2,3>& b, const Vec<T3,3>& c) -> decltype(a[0] * b[0] * c[0]) {
  return dot(a, cross(b, c));
}

/// Returns a Vec from the vector triple product a x (b x c)
template <typename T1, typename T2, typename T3>
inline constexpr auto vector_triple_product(const Vec<T1,3>& a, const Vec<T2,3>& b, const Vec<T3,3>& c) -> Vec<decltype(a[0] * b[0] * c[0]), 3> {
  return cross(a, cross(b, c));
}

/// Returns a Vec with the element wise multiplication of a and b,
/// c_k = a_k * b_k
template <typename T1, typename T2>
inline constexpr auto hadamard_product(const Vec<T1,3>& a, const Vec<T2,3>& b) -> Vec<decltype(a[0] * b[0]), 3> {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

/// Returns the angle in radians between vector a and b
template <typename T>
inline constexpr T angle(const Vec<T,3>& a, const Vec<T,3>& b) {
  return acos(dot(a,b) / (norm(a), norm(b)));
}

/// Returns a Vec3 in cartesian coordinates (x, y, z) from the polar coordinates
/// (r, theta, phi), where theta is the polar angle (from z) and phi is the
/// azimuthal angle (x-y plane, from x). Angles must be in radians.
inline Vec3 spherical_to_cartesian_vector(const double r, const double theta, const double phi) {
  return {r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)};
}

/// Returns the polar angle in radians of the cartesian vector a. The polar
/// angle is the angle down from the +z axis.
inline double polar_angle(const Vec<double,3> a) {
  return acos(a[2]/norm(a));
}

/// Returns the azimuthal angle in radians of the cartesian vector a. The
/// azimuthal angle is the angle in the x-y plane starting from x.
inline double azimuthal_angle(const Vec<double,3> a) {
  return atan2(a[1], a[0]);
}

/// Returns a unit vector by performing a / |a|. If |a| = 0 the function performs
/// a zero division and the results will be +infinity.
template <typename T>
inline auto normalize(const Vec<T,3>& a) -> Vec<decltype(a[0] / std::abs(a[0])), 3> {
  return a / norm(a);
}

/// Returns true if all components of a are approximately zero (using a relative
/// epsilon), false otherwise.
template <typename T>
inline bool approximately_zero(const Vec<T,3>& a, const T& epsilon) {
  for (auto n = 0; n < 3; ++n) {
    if (!approximately_zero(a[n], epsilon)) {
      return false;
    }
  }
  return true;
}

/// Performs a safe floating point comparison between vectors a and b to check
/// for equality. Returns true if all components of a are approximately equal to
/// b (using a relative epsilon), false otherwise.
template <typename T>
inline bool approximately_equal(const Vec<T,3>& a, const Vec<T,3>& b, const T& epsilon) {
  for (auto n = 0; n < 3; ++n) {
    if (!approximately_equal(a[n], b[n], epsilon)) {
      return false;
    }
  }
  return true;
}

// Returns the unit vector of `a`. For vectors with a length less than a small
// value `epsilon` the original vector is returned.
template <typename T>
inline auto unit_vector(const Vec<T, 3> &a, const T& epsilon = DBL_EPSILON) -> Vec<decltype(a[0] / std::abs(a[0])), 3> {
  const auto length = norm(a);
  if (approximately_zero(length, epsilon)) {
    return a;
  }

  return a / length;
}

/// Returns the sum of the elements in the vector a
template <typename T>
inline constexpr T sum(const Vec<T,3>& a) {
  return a[0] + a[1] + a[2];
}

/// Returns the product of the elements in the vector a
template <typename T>
inline constexpr T product(const Vec<T,3>& a) {
  return a[0] * a[1] * a[2];
}

/// Returns a complex Vec3 with the conjugate of each component of a,
/// x_k = conj(a_k)
template <typename T>
inline constexpr Vec<std::complex<T>,3> conj(const Vec<std::complex<T>,3>& a) {
  return {std::conj(a[0]), std::conj(a[1]), std::conj(a[2])};
}

/// Returns a complex Vec3 with each component of a truncated,
/// x_k = trunc(a_k)
template <typename T>
inline constexpr Vec<T,3> trunc(const Vec<T,3>& a) {
  return {std::trunc(a[0]), std::trunc(a[1]), std::trunc(a[2])};
}

/// Returns a Vec3 of doubles static_casted from the vector a
template <typename T>
inline constexpr Vec<double,3> to_double(const Vec<T,3>& a) {
  return {
    static_cast<double>(a[0]),
    static_cast<double>(a[1]),
    static_cast<double>(a[2])
  };
}

/// Returns a Vec3 of ints static_casted from the vector a
template <typename T>
inline constexpr Vec<int,3> to_int(const Vec<T,3>& a) {
  return {
      static_cast<int>(a[0]),
      static_cast<int>(a[1]),
      static_cast<int>(a[2])
  };
}

/// Returns the largest absolute value in the array
template <typename T, std::size_t N>
T absolute_max(const std::array<T,N>& x) {
  return std::abs(*std::max_element(x.begin(), x.end(),
                           [](const T& a, const T& b) {
                               return std::abs(a) < std::abs(b); }));
}

/// Returns a Vec with each component divided by its magnitude. Components
/// of zero are left as zero. x_k = a_k / |a_k|
template <typename T, std::size_t N>
Vec<T, N> normalize_components(const Vec<T, N>& a) {
  Vec<T, N> result;
  for (auto i = 0; i < N; ++i) {
    a[i] != 0 ? result[i] = a[i]/abs(a[i]) : result[i] = 0;
  }
  return result;
}


template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Vec<T,3> &a) {
  auto w = os.width();
  auto p = os.precision();
  os << std::right << std::setprecision(p) << std::setw(w) << a[0] << " ";
  os << std::right << std::setprecision(p) << std::setw(w) << a[1] << " ";
  os << std::right << std::setprecision(p) << std::setw(w) << a[2];
  return os;
}


#endif //JAMS_VEC3_H
