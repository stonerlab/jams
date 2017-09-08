//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_VEC3_H
#define JAMS_VEC3_H

#include <array>

template <typename T, std::size_t N>
using Vec = std::array<T, N>;

#include <cmath>
#include <iostream>

#include "mat3.h"

template <typename T>
inline T dot(const Vec<T,3>& a, const Vec<T,3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
inline T product(const Vec<T,3>& a) {
  return a[0] * a[1] * a[2];
}

template <typename T>
inline Vec<T,3> cross(const Vec<T,3>& a, const Vec<T,3>& b) {
  return {a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]};
}

template <typename T>
inline T abs_sq(const Vec<T,3>& a) {
  return dot(a, a);
}

template <typename T>
inline double abs(const Vec<T,3>& a) {
  return sqrt(dot(a, a));
}

template <typename T>
inline Vec<T,3> trunc(const Vec<T,3>& a) {
  return {trunc(a[0]), trunc(a[1]), trunc(a[2])};
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Vec<T,3> &a)
{
  return os << a[0] << "\t" << a[1] << "\t" << a[2];
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
inline Vec<T,3> operator+=(Vec<T,3>& lhs, const T& rhs) {
  return {lhs[0] += rhs, lhs[1] += rhs, lhs[2] += rhs};
}

template <typename T>
inline Vec<T,3> operator+=(Vec<T,3>& lhs, const Vec<T,3>& rhs) {
  return {lhs[0] += rhs[0], lhs[1] += rhs[1], lhs[2] += rhs[2]};
}

template <typename T>
inline Vec<T,3> operator/=(Vec<T,3>& lhs, const T& rhs) {
  return {lhs[0] /= rhs, lhs[1] /= rhs, lhs[2] /= rhs};
}

template <typename T>
inline Vec<T,3> normalize(const Vec<T,3>& a) {
  return a / abs(a);
}

//template <typename T>
//inline Vec<T,3> operator*(const Mat3<T>& lhs, const Vec<T,3>& rhs) {
//  Vec<T,3> result = {0.0, 0.0, 0.0};
//  for (auto i = 0; i < 3; ++i) {
//    for (auto j = 0; j < 3; ++j) {
//      result[i] += lhs[i][j] * rhs[j];
//    }
//  }
//  return result;
//}


#endif //JAMS_VEC3_H
