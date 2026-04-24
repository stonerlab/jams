//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_MAT3_H
#define JAMS_MAT3_H

#include <cassert>
#include <array>
#include <jams/interface/blas.h>
#include <limits>
#include "jams/containers/mat.h"
#include "jams/containers/vec3.h"

const Mat<double, 3, 3> kIdentityMat3 = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
const Mat<std::complex<double>, 3, 3> kIdentityMat3cx = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
const Mat<double, 3, 3> kZeroMat3 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

const Mat<jams::Real, 3, 3> kIdentityMat3R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
const Mat<jams::Real, 3, 3> kZeroMat3R = {0, 0, 0, 0, 0, 0, 0, 0, 0};

// Vec<double, 3> specialization
template <typename T>
inline bool approximately_equal(const Mat<T,3,3>& a, const Mat<T,3,3>& b, const T& epsilon) {
  for (auto m = 0; m < 3; ++m) {
    for (auto n = 0; n < 3; ++n) {
      if (!approximately_equal(a[m][n], b[m][n], epsilon)) {
        return false;
      }
    }
  }
  return true;
}

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
T determinant(const Mat<T,3,3>& a);

template <>
inline float determinant(const Mat<float, 3, 3>& a) {
  int n = 3;
  int lda = 3;
  int ipiv[3];
  int info;

  float A_copy[9] = {
      a[0][0], a[1][0], a[2][0],
      a[0][1], a[1][1], a[2][1],
      a[0][2], a[1][2], a[2][2]
  };

  sgetrf_(&n, &n, A_copy, &lda, ipiv, &info);
  assert(info == 0);

  float det = 1.0f;
  for (int i = 0; i < n; ++i) {
    det *= A_copy[i * n + i];
  }

  int num_swaps = 0;
  for (int i = 0; i < n; ++i) {
    if (ipiv[i] != i + 1) ++num_swaps;
  }
  
  if (num_swaps % 2 != 0) {
    det = -det;
  }

  return det;
}

template <>
inline double determinant(const Mat<double, 3, 3>& a) {
  int n = 3;
  int lda = 3;
  int ipiv[3];
  int info;

  double A_copy[9] = {
      a[0][0], a[1][0], a[2][0],
      a[0][1], a[1][1], a[2][1],
      a[0][2], a[1][2], a[2][2]
  };

  dgetrf_(&n, &n, A_copy, &lda, ipiv, &info);
  assert(info == 0);

  double det = 1.0;
  for (int i = 0; i < n; ++i) {
    det *= A_copy[i * n + i];
  }

  int num_swaps = 0;
  for (int i = 0; i < n; ++i) {
    if (ipiv[i] != i + 1) {
      ++num_swaps;
    }
  }

  if (num_swaps % 2 != 0) {
    det = -det;
  }

  return det;
}

template <typename T>
Mat<T, 3, 3> inverse(const Mat<T, 3, 3>& a);

template <>
inline Mat<float, 3, 3> inverse(const Mat<float, 3, 3>& a) {
  int n = 3;
  int lda = 3;
  int ipiv[3];
  int info;

  float A_copy[9] = {
      a[0][0], a[1][0], a[2][0],
      a[0][1], a[1][1], a[2][1],
      a[0][2], a[1][2], a[2][2]
  };

  sgetrf_(&n, &n, A_copy, &lda, ipiv, &info);
  assert(info == 0);

  float work[64];
  int lwork = 64;
  sgetri_(&n, A_copy, &lda, ipiv, work, &lwork, &info);
  assert(info == 0);

  return {
      A_copy[0], A_copy[3], A_copy[6],
      A_copy[1], A_copy[4], A_copy[7],
      A_copy[2], A_copy[5], A_copy[8]
  };
}

template <>
inline Mat<double, 3, 3> inverse(const Mat<double, 3, 3>& a) {
  int n = 3;
  int lda = 3;
  int ipiv[3];
  int info;

  double A_copy[9] = {
      a[0][0], a[1][0], a[2][0],
      a[0][1], a[1][1], a[2][1],
      a[0][2], a[1][2], a[2][2]
  };

  dgetrf_(&n, &n, A_copy, &lda, ipiv, &info);
  assert(info == 0);

  double work[64];
  int lwork = 64;
  dgetri_(&n, A_copy, &lda, ipiv, work, &lwork, &info);
  assert(info == 0);

  return {
      A_copy[0], A_copy[3], A_copy[6],
      A_copy[1], A_copy[4], A_copy[7],
      A_copy[2], A_copy[5], A_copy[8]
  };
}

inline Mat<double, 3, 3> rotation_matrix_y(const double& theta) {
  return {
      cos(theta),  0.0, sin(theta),
      0.0,         1.0,        0.0,
      -sin(theta), 0.0, cos(theta)
  };
}

inline Mat<double, 3, 3> rotation_matrix_z(const double& phi) {
  return {
      cos(phi),  -sin(phi), 0.0,
      sin(phi),   cos(phi), 0.0,
      0.0,        0.0,      1.0
  };
}

inline Mat<double, 3, 3> rotation_matrix_yz(const double theta, const double phi) {
  const double c_t = cos(theta);
  const double c_p = cos(phi);
  const double s_t = sin(theta);
  const double s_p = sin(phi);

  return Mat<double, 3, 3> {c_t*c_p, -c_t*s_p, s_t, s_p, c_p, 0, -c_p*s_t, s_t*s_p, c_t};
}

template <typename T>
inline T max_abs(const Mat<T,3,3>& a) {
  T max = 0.0;
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      if (std::abs(a[i][j]) > max) {
        max = std::abs(a[i][j]);
      }
    }
  }
  return max;
}

/// Returns rotation matrix which rotates by an angle theta around the given axis
/// See: https://en.wikipedia.org/wiki/Rotation_matrix
inline Mat<double, 3, 3> rotation_matrix(const Vec<double, 3>& axis, const double theta) {
  // make sure we have a unit vector
  const Vec<double, 3> u = jams::unit_vector(axis);
  const double c = cos(theta);
  const double s = sin(theta);
  return {
    c + u[0]*u[0]*(1-c),       u[0]*u[1]*(1-c) - u[2]*s,  u[0]*u[2]*(1-c) + u[1]*s,
    u[1]*u[0]*(1-c) + u[2]*s,  c + u[1]*u[1]*(1-c),       u[1]*u[2]*(1-c) - u[0]*s,
    u[2]*u[0]*(1-c) - u[1]*s,  u[2]*u[1]*(1-c) + u[0]*s,  c + u[2]*u[2]*(1-c) };
}


inline Mat<double, 3, 3> rotation_matrix_from_axis_angle(const Vec<double, 3>& axis, double angle) {
  constexpr double eps_axis = 1e-14;

  const double n = jams::norm(axis);
  if (n < eps_axis) return kIdentityMat3;

  const Vec<double, 3> u = axis / n;

  double s = std::sin(angle);
  double c1m = jams::cos1m(angle);

  const Mat<double, 3, 3> vx = ssc(u);
  const Mat<double, 3, 3> vx2 = vx * vx;

  Mat<double, 3, 3> R = kIdentityMat3 + s * vx + c1m * vx2;

#ifndef NDEBUG
  const Mat<double, 3, 3> RtR = transpose(R) * R;
  assert(approximately_equal(RtR, kIdentityMat3, 1e-10));
  assert(std::abs(determinant(R) - 1.0) < 1e-10);
#endif

  return R;
}

// calculates a rotation matrix from Vec<double, 3> a to Vec<double, 3> b
inline Mat<double, 3, 3> rotation_matrix_between_vectors(const Vec<double, 3>& a, const Vec<double, 3>& b) {
  constexpr double eps = 1e-14;      // for norms
  constexpr double eps_s2 = 1e-24;   // for sin^2

  const double na = jams::norm(a), nb = jams::norm(b);
  if (na < eps || nb < eps) return kIdentityMat3;

  const Vec<double, 3> ua = a / na;
  const Vec<double, 3> ub = b / nb;

  const double c_raw = jams::dot(ua, ub);
  const double c = std::max(-1.0, std::min(1.0, c_raw));

  Vec<double, 3> v = jams::cross(ua, ub);
  const double s2 = jams::dot(v, v);

  if (s2 < eps_s2) {
    if (c > 0.0) return kIdentityMat3; // parallel
    // antiparallel: pick stable orthogonal axis
    Vec<double, 3> ortho;
    if (std::abs(ua[0]) <= std::abs(ua[1]) && std::abs(ua[0]) <= std::abs(ua[2])) ortho = {1,0,0};
    else if (std::abs(ua[1]) <= std::abs(ua[2])) ortho = {0,1,0};
    else ortho = {0,0,1};
    Vec<double, 3> axis = jams::unit_vector(jams::cross(ua, ortho));
    return rotation_matrix_from_axis_angle(axis, kPi);
  }

  const double s = std::sqrt(s2);
  const double theta = std::atan2(s, c);
  const Vec<double, 3> axis = v / s;
  return rotation_matrix_from_axis_angle(axis, theta);
}

#endif //JAMS_MAT3_H
