//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_VEC3_H
#define JAMS_VEC3_H

#include <array>
#include <cmath>
#include <iosfwd>
#include <iomanip>

template <typename T, std::size_t N>
using Vec = std::array<T, N>;

using Vec3  = std::array<double, 3>;
using Vec3f = std::array<float, 3>;
using Vec3b = std::array<bool, 3>;
using Vec3i = std::array<int, 3>;

using Vec4  = std::array<double, 4>;
using Vec4i = std::array<int, 4>;

template <typename T>
inline Vec<T,3> operator-(const Vec<T,3>& rhs) {
  return {-rhs[0], -rhs[1], -rhs[2]};
}

template <typename T1, typename T2>
inline auto operator*(const T1& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs * rhs[0]), 3> {
  return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
}

template <typename T1, typename T2>
inline auto operator*(const Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] * rhs), 3> {
  return {lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs};
}

template <typename T1, typename T2>
inline auto operator/(const Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] / rhs), 3> {
  return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs};
}

template <typename T1, typename T2>
inline auto operator+(const Vec<T1,3>& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs[0] + rhs[0]), 3> {
  return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

template <typename T1, typename T2>
inline auto operator-(const Vec<T1,3>& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs[0] - rhs[0]), 3> {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

template <typename T1, typename T2>
inline auto operator+=(Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] + rhs), 3> {
  return {lhs[0] += rhs, lhs[1] += rhs, lhs[2] += rhs};
}

template <typename T1, typename T2>
inline auto operator+=(Vec<T1,3>& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs[0] + rhs[0]), 3> {
  return {lhs[0] += rhs[0], lhs[1] += rhs[1], lhs[2] += rhs[2]};
}

template <typename T1, typename T2>
inline auto operator-=(Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] - rhs), 3> {
  return {lhs[0] -= rhs, lhs[1] -= rhs, lhs[2] -= rhs};
}

template <typename T1, typename T2>
inline auto operator-=(Vec<T1,3>& lhs, const Vec<T2,3>& rhs) -> Vec<decltype(lhs[0] - rhs[0]), 3> {
  return {lhs[0] -= rhs[0], lhs[1] -= rhs[1], lhs[2] -= rhs[2]};
}

template <typename T1, typename T2>
inline auto operator*=(Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] * rhs), 3> {
  return {lhs[0] *= rhs, lhs[1] *= rhs, lhs[2] *= rhs};
}

template <typename T1, typename T2>
inline auto operator/=(Vec<T1,3>& lhs, const T2& rhs) -> Vec<decltype(lhs[0] / rhs), 3> {
  return {lhs[0] /= rhs, lhs[1] /= rhs, lhs[2] /= rhs};
}

template <typename T>
inline bool equal(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]);
}

template <typename T>
inline bool operator==(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return equal(lhs, rhs);
}

template <typename T>
inline bool operator!=(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return !equal(lhs, rhs);
}

template <typename T>
inline auto operator%(const Vec<T,3>& lhs, const Vec<T,3>& rhs) -> Vec<decltype(lhs[0] % rhs[0]), 3> {
  return {lhs[0] % rhs[0], lhs[1] % rhs[1], lhs[2] % rhs[2]};
}

template <typename T1, typename T2>
inline auto dot(const Vec<T1,3>& a, const Vec<T2,3>& b) -> decltype(a[0] * b[0]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T1, typename T2>
inline auto cross(const Vec<T1,3>& a, const Vec<T2,3>& b) -> Vec<decltype(a[0] * b[0]), 3> {
  return {a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]};
}

template <typename T1, typename T2, typename T3>
inline auto scalar_triple_product(const Vec<T1,3>& a, const Vec<T2,3>& b, const Vec<T3,3>& c) -> decltype(a[0] * b[0] * c[0]) {
  return dot(a, cross(b, c));
}

template <typename T1, typename T2, typename T3>
inline auto vector_triple_product(const Vec<T1,3>& a, const Vec<T2,3>& b, const Vec<T3,3>& c) -> Vec<decltype(a[0] * b[0] * c[0]), 3> {
  return cross(a, cross(b, c));
}

template <typename T1, typename T2>
inline auto scale(const Vec<T1,3>& a, const Vec<T2,3>& b) -> Vec<decltype(a[0] * b[0]), 3> {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

template <typename T>
inline auto abs(const Vec<T,3>& a) -> decltype(std::sqrt(a[0])) {
  return std::sqrt(dot(a, a));
}

template <typename T>
inline T abs_sq(const Vec<T,3>& a) {
  return dot(a, a);
}

template <typename T>
inline auto normalize(const Vec<T,3>& a) -> Vec<decltype(a[0] / std::abs(a[0])), 3> {
  return a / abs(a);
}

template <typename T>
inline T sum(const Vec<T,3>& a) {
  return a[0] + a[1] + a[2];
}

template <typename T>
inline T product(const Vec<T,3>& a) {
  return a[0] * a[1] * a[2];
}

template <typename T>
inline Vec<T,3> trunc(const Vec<T,3>& a) {
  return {std::trunc(a[0]), std::trunc(a[1]), std::trunc(a[2])};
}

template <typename T>
inline Vec<double,3> to_double(const Vec<T,3>& a) {
  return {
    static_cast<double>(std::trunc(a[0])),
    static_cast<double>(std::trunc(a[1])),
    static_cast<double>(std::trunc(a[2]))
  };
}


template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Vec<T,3> &a) {
  return os << a[0] << " " << a[1] << " " << a[2];
}


#endif //JAMS_VEC3_H
