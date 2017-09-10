//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_MAT3_H
#define JAMS_MAT3_H

#include <array>
#include <limits>
#include "vec3.h"

template <typename T, std::size_t M, std::size_t N>
using Mat = std::array<std::array<T, M>, N>;

const Mat<double, 3, 3> kIdentityMat3 = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

template <typename T>
Mat<T,3,3> matrix_from_rows(const Vec<T,3>& a, const Vec<T,3>& b, const Vec<T,3>& c) {
  return {a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]};
}

template <typename T>
Mat<T,3,3> matrix_from_cols(const Vec<T,3>& a, const Vec<T,3>& b, const Vec<T,3>& c) {
  return {a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]};
}

template <typename T>
Mat<T,3,3> ssc(const Vec<T,3> &v) {
  // skew symmetric cross product matrix
  return {0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0};
}

template <typename T>
T determinant(const Mat<T,3,3>& a) {
  return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
         +a[0][1]*(a[1][2]*a[2][0]-a[1][0]*a[2][2])
         +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
}

template <typename T>
Mat<T,3,3> inverse(const Mat<T,3,3>& a) {
  T det = 1.0/determinant(a);

  Mat<T,3,3> out;
  out[0][0] = det*(a[1][1]*a[2][2]-a[1][2]*a[2][1]);
  out[0][1] = det*(a[2][1]*a[0][2]-a[0][1]*a[2][2]);
  out[0][2] = det*(a[0][1]*a[1][2]-a[1][1]*a[0][2]);

  out[1][0] = det*(a[1][2]*a[2][0]-a[1][0]*a[2][2]);
  out[1][1] = det*(a[0][0]*a[2][2]-a[0][2]*a[2][0]);
  out[1][2] = det*(a[0][2]*a[1][0]-a[0][0]*a[1][2]);

  out[2][0] = det*(a[1][0]*a[2][1]-a[2][0]*a[1][1]);
  out[2][1] = det*(a[2][0]*a[0][1]-a[0][0]*a[2][1]);
  out[2][2] = det*(a[0][0]*a[1][1]-a[1][0]*a[0][1]);

  return out;
}

template <typename T>
Mat<T,3,3> transpose(const Mat<T,3,3>& a) {
  return {
    a[0][0], a[1][0], a[2][0],
    a[0][1], a[1][1], a[2][1],
    a[0][2], a[1][2], a[2][2]
  };
}


template <typename T>
inline Vec<T,3> operator*(const Mat<T,3,3>& lhs, const Vec<T,3>& rhs) {
  return {
          lhs[0][0] * rhs[0] + lhs[0][1] * rhs[1] + lhs[0][2] * rhs[2],
          lhs[1][0] * rhs[0] + lhs[1][1] * rhs[1] + lhs[1][2] * rhs[2],
          lhs[2][0] * rhs[0] + lhs[2][1] * rhs[1] + lhs[2][2] * rhs[2]
  };
}

template <typename T>
inline Mat<T,3,3> operator*(const T& lhs, const Mat<T,3,3>& rhs) {
  return { lhs * rhs[0][0], lhs * rhs[0][1], lhs * rhs[0][2],
           lhs * rhs[1][0], lhs * rhs[1][1], lhs * rhs[1][2],
           lhs * rhs[2][0], lhs * rhs[2][1], lhs * rhs[2][2]
  };
}

template <typename T>
inline Mat<T,3,3> operator/(const Mat<T,3,3>& lhs, const T& rhs) {
  return {lhs[0][0] / rhs, lhs[0][1] / rhs, lhs[0][2] / rhs,
          lhs[1][0] / rhs, lhs[1][1] / rhs, lhs[1][2] / rhs,
          lhs[2][0] / rhs, lhs[2][1] / rhs, lhs[2][2] / rhs
  };
}

template <typename T>
inline Mat<T,3,3> operator*(const Mat<T,3,3>& lhs, const Mat<T,3,3>& rhs) {
  Mat<T,3,3> result = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      for (auto k = 0; k < 3; ++k) {
        result[i][j] += lhs[i][k] * rhs[k][j];
      }
    }
  }
  return result;
}

template <typename T>
inline T max_norm(const Mat<T,3,3>& a) {
  T max = std::numeric_limits<T>::min();
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      if (std::abs(a[i][j]) > max) {
        max = std::abs(a[i][j]);
      }
    }
  }
  return max;
}

#endif //JAMS_MAT3_H
