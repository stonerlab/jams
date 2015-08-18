// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_MATHS_H
#define JAMS_CORE_MATHS_H

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "core/consts.h"

inline double square(const double &x) {
  return x*x;
}

inline int nint(const double &x) {
  return floor(x+0.5);
}

inline double deg_to_rad(const double &angle) {
  return angle*(kPi/180.0);
}

inline double rad_to_deg(const double &angle) {
  return angle*(180.0/kPi);
}

// greatest common divisor
long gcd(long a, long b);

// lowest common multiple
long lcm(long a, long b);

///
/// @brief  Approximates real as an integer fraction top / bottom
///
/// Comes from here https://www.ics.uci.edu/~eppstein/numth/frap.c
///
/// @param[out]  nominator of fraction
/// @param[out]  denominator of fraction
/// @param[in]   real number to approximate as fraction
/// @param[in]   max_denomiantor to search
/// @return error of the fraction relative to the float
///
double approximate_float_as_fraction(long &nominator, long &denominator, const double real, const long max_denomiantor);

///
/// @brief  Sign transfer function from y->x
///
/// The sign transfer function transfers the sign of argument y to the
/// value of x. This is defined as for the Fortran function of the same
/// name.
///
/// @param[in]  x value to transfer to
/// @param[in]  y value to take sign from
/// @return x with the sign of y
///
template <typename _Tp1, typename _Tp2>
inline _Tp1 sign(const _Tp1 &x, const _Tp2 &y) {
  if (y >= 0.0) {
    return std::abs(x);
  } else {
    return -std::abs(x);
  }
}

///
/// @brief  Returns sign of argument
///
/// Instead of sign transfer this just returns the sign of the argument
/// Source: https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
///
/// @param[in]  x value to take sign from
/// @return -1, 0, 1 depending on sign
///
template <typename _Tp>
inline int sgn(_Tp x) {
    return (_Tp(0) < x) - (x < _Tp(0));
}

///
/// @brief  Generates next point space symmetry representation
///
/// This function generates the next point space symmetry representation from
/// a set of points, where the ascending positive valued points are the lowest
/// configuration and the negative descending the highest.
///
/// @param  [in, out] pts[3] points in space
/// @return false if no more symmetry points can be generated, otherwise true
///
template <typename _Tp>
bool next_point_symmetry(_Tp pts[3]) {
  // check number is valid (0 cannot be negative)
  bool valid = false;

  // extract sign mask
  int sgn[3] = { sign(1, pts[0]), sign(1, pts[1]), sign(1, pts[2]) };

  // absolute value of the points
  _Tp abspts[3] = { std::abs(pts[0]), std::abs(pts[1]), std::abs(pts[2])};

  // loop until number are valied wrt 0
  do {
    // permute the absolute values
    if (std::next_permutation(abspts, abspts+3)) {
      pts[0] = abspts[0]*sgn[0];
      pts[1] = abspts[1]*sgn[1];
      pts[2] = abspts[2]*sgn[2];

    // if no more permutation is possible then permute the signs
    } else {
      // once -1, -1, -1 is reached, all sgnmetries have been computed
      if (sgn[0]+sgn[1]+sgn[2] == -3) {
        return false;
      }

      // re-sort absolute values to ascend for permutation
      std::sort(abspts, abspts+3);

      // permute signs if possible
      if (std::next_permutation(sgn, sgn+3)) {
        pts[0] = abspts[0]*sgn[0];
        pts[1] = abspts[1]*sgn[1];
        pts[2] = abspts[2]*sgn[2];
      } else {
        // sort signs so that sgn[2] is 1 if possible
        std::sort(sgn, sgn+3);
        sgn[2] = -1;
        // re-sort so that numbers are ascending for permutation
        std::sort(sgn, sgn+3);

        pts[0] = abspts[0]*sgn[0];
        pts[1] = abspts[1]*sgn[1];
        pts[2] = abspts[2]*sgn[2];
      }
    }

    // check validity of values with respect to zero signs
    valid = true;
    for (int i = 0; i < 3; ++i) {
      if (pts[i] == 0 && sgn[i] == -1) {
        valid = false;
      }
    }
  } while (valid == false);

  return true;
}

inline void cartesian_to_spherical(const double x,
    const double y, const double z, double* r, double* theta, double* phi) {
  (*r) = sqrt(x*x+y*y+z*z);
  (*theta) = acos(z/(*r));
  (*phi) = atan2(y, x);
}

inline void spherical_to_cartesian(const double r,
    const double theta, const double phi, double* x, double* y, double* z) {
  (*x) = r*cos(theta)*cos(phi);
  (*y) = r*cos(theta)*sin(phi);
  (*z) = r*sin(theta);
}

void matrix_invert(const double in[3][3], double out[3][3]);

template <typename _Tp>
void matmul(const _Tp a[3][3], const _Tp b[3][3], _Tp c[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = 0;
      for (int k = 0; k < 3; ++k) {
#ifdef FP_FAST_FMA
        c[i][j] = fma(a[i][k], b[k][j], c[i][j]);
#else
        c[i][j] += a[i][k]*b[k][j];
#endif
      }
    }
  }
}

template <typename _Tp>
void matmul(const _Tp a[3][3], const _Tp x[3], _Tp y[3]) {
  int i, j;
  for (i = 0; i < 3; ++i) {
    y[i] = 0;
    for (j = 0; j < 3; ++j) {
#ifdef FP_FAST_FMA
      y[i] = fma(a[i][j], x[j], y[i]);
#else
      y[i] += a[i][j]*x[j];
#endif
    }
  }
}

template <typename _A, typename _B, typename _C>
void matmul(const _A a[3][3], const _B x[3], _C y[3]) {
  int i, j;
  for (i = 0; i < 3; ++i) {
    y[i] = 0;
    for (j = 0; j < 3; ++j) {
#ifdef FP_FAST_FMA
      y[i] = fma(a[i][j], x[j], y[i]);
#else
      y[i] += a[i][j]*x[j];
#endif
    }
  }
}

template <typename _Tp>
inline _Tp DotProduct(const _Tp a[3], const _Tp b[3]) {
#ifdef FP_FAST_FMA
  return fma(a[2], b[2], fma(a[1], b[1], a[0]*b[0]));
#else
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
#endif
}

template <typename _Tp>
inline void CrossProduct(const _Tp a[3], const _Tp b[3], _Tp out[3]) {
  out[0] = a[1]*b[2] - a[2]*b[1];
  out[1] = a[2]*b[0] - a[0]*b[2];
  out[2] = a[0]*b[1] - a[1]*b[0];
}

#endif  // JAMS_CORE_MATHS_H
