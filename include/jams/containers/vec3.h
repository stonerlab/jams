//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_VEC3_H
#define JAMS_VEC3_H

#include <array>
#include <cmath>
#include <iosfwd>

template <typename T, std::size_t N>
using Vec = std::array<T, N>;

template <typename T>
inline Vec<T,3> operator-(const Vec<T,3>& rhs) {
  return {-rhs[0], -rhs[1], -rhs[2]};
}

template <typename T>
inline Vec<T,3> operator*(const T& lhs, const Vec<T,3>& rhs) {
  return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
}

template <typename T>
inline Vec<T,3> operator*(const Vec<T,3>& lhs, const T& rhs) {
  return {rhs * lhs[0], rhs * lhs[1], rhs * lhs[2]};
}

template <typename T>
inline Vec<T,3> operator/(const Vec<T,3>& lhs, const T& rhs) {
  return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs};
}

template <typename T>
inline Vec<T,3> operator+(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

template <typename T>
inline Vec<T,3> operator-(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

template <typename T>
inline Vec<T,3> operator+=(Vec<T,3>& lhs, const T& rhs) {
  return {lhs[0] += rhs, lhs[1] += rhs, lhs[2] += rhs};
}

template <typename T>
inline Vec<T,3> operator+=(Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return {lhs[0] += rhs[0], lhs[1] += rhs[1], lhs[2] += rhs[2]};
}

template <typename T>
inline Vec<T,3> operator-=(Vec<T,3>& lhs, const T& rhs) {
  return {lhs[0] -= rhs, lhs[1] -= rhs, lhs[2] -= rhs};
}

template <typename T>
inline Vec<T,3> operator-=(Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return {lhs[0] -= rhs[0], lhs[1] -= rhs[1], lhs[2] -= rhs[2]};
}

template <typename T>
inline Vec<T,3> operator*=(Vec<T,3>& lhs, const T& rhs) {
  return {lhs[0] *= rhs, lhs[1] *= rhs, lhs[2] *= rhs};
}

template <typename T>
inline Vec<T,3> operator/=(Vec<T,3>& lhs, const T& rhs) {
  return {lhs[0] /= rhs, lhs[1] /= rhs, lhs[2] /= rhs};
}

template <typename T>
inline bool equal(const Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return (lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]);
}

template <typename T>
inline bool equal(const Vec<T,3>& lhs, const Vec<T,3>& rhs, const double eps) {
  for (auto n = 0; n < 3; ++n) {
    if (std::abs(lhs[n] - rhs[n]) > eps) {
      return false;
    }
  }
  return true;
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
inline T dot(const Vec<T,3>& a, const Vec<T,3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
inline Vec<T,3> cross(const Vec<T,3>& a, const Vec<T,3>& b) {
  return {a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]};
}

template <typename T>
inline double abs(const Vec<T,3>& a) {
  return sqrt(dot(a, a));
}

template <typename T>
inline T abs_sq(const Vec<T,3>& a) {
  return dot(a, a);
}

template <typename T>
inline Vec<T,3> normalize(const Vec<T,3>& a) {
  return a / abs(a);
}

template <typename T>
inline Vec<T,3> scale(const Vec<T,3>& a, const Vec<T,3>& b) {
  return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
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
inline std::ostream& operator<<(std::ostream& os, const Vec<T,3> &a) {
  return os << a[0] << " " << a[1] << " " << a[2];
}


#endif //JAMS_VEC3_H
