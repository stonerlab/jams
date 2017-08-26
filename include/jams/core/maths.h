// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_MATHS_H
#define JAMS_CORE_MATHS_H

#include <complex>
#include <algorithm>
#include <cmath>

#include "jams/core/consts.h"

#include "jams/containers/vec3.h"
#include "jams/containers/mat3.h"
#include "jams/core/types.h"

using std::pow;

inline double square(const double &x) {
  return x*x;
}

inline int nint(const double &x) {
  return floor(x+0.5);
}

inline double gaussian(const double x, const double sigma, const double mean = 0.0) {
  return kOne_SqrtTwoPi*exp(-0.5*std::pow((x - mean) / sigma, 2))/sigma;
}

inline double deg_to_rad(const double &angle) {
  return angle*(kPi/180.0);
}

inline double rad_to_deg(const double &angle) {
  return angle*(180.0/kPi);
}

inline double azimuthal_angle(const Vec3 a) {
  return acos(a[2]/abs(a));
}

inline double polar_angle(const Vec3 a) {
  return atan2(a[2], a[0]);
}

inline double azimuthal_angle(const double a[3]) {
  return acos(a[2]/sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]));
}

inline double polar_angle(const double a[3]) {
  return atan2(a[1], a[0]);
}

inline double azimuthal_angle(const double x, const double y, const double z) {
  return acos(z/sqrt(x*x + y*y + z*z));
}

inline double polar_angle(const double x, const double y, const double z) {
  return atan2(y, x);
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

template <typename _Tp1, typename _Tp2>
inline bool same_sign(const _Tp1 x, const _Tp2 y) {
  return (x >= 0) ^ (y < 0);
}

inline bool floats_are_equal(const double x, const double y, const double tolerance = 1e-8) {
  using std::abs;
  return same_sign(x, y) && (abs(x - y) < tolerance);
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

inline Vec3 spherical_to_cartesian_vector(const double r,
    const double theta, const double phi) {
  return {r*cos(theta)*cos(phi), r*cos(theta)*sin(phi), r*sin(theta)};
}

void matrix_invert(const double in[3][3], double out[3][3]);
void matrix_invert(const float in[3][3], float out[3][3]);

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

Mat3 rotation_matrix_yz(const double theta, const double phi);
Mat3 rotation_matrix_between_vectors(const Vec3 &x, const Vec3 &y);


// Legendre polynomials
double legendre_poly(const double x, const int n);

// differentiated Legendre polynomials
double legendre_dpoly(const double x, const int n);



// uniaxial anisotropy is an expansion in legendre polynomials so lets define upto n = 6
// so we have some fast intrinsics

inline double legendre_poly_0(const double x) {
  return 1.0;
}

inline double legendre_poly_1(const double x) {
  return x;
}

inline double legendre_poly_2(const double x) {
  // (3x^2 - 1)/2
  return (1.5 * x * x - 0.5);
}

inline double legendre_poly_3(const double x) {
  // (5x^3 - 3x)/2
  return (2.5 * x * x * x - 1.5 * x);
}

inline double legendre_poly_4(const double x) {
  // (35x^4 - 30x^2 + 3)/8
  return (4.375 * x * x * x * x - 3.75 * x * x + 0.375);
}

inline double legendre_poly_5(const double x) {
  // (63x^5 - 70x^3 + 15x)/8
  return (7.875 * x * x * x * x * x - 8.75 * x * x * x + 1.875 * x);
}

inline double legendre_poly_6(const double x) {
  // (231x^6 - 315x^4 + 105x^2 - 5)/16
  return (14.4375 * x * x * x * x * x * x - 19.6875 * x * x * x * x + 6.5625 * x * x - 0.3125);
}

inline double legendre_dpoly_0(const double x) {
  return 0.0;
}

inline double legendre_dpoly_1(const double x) {
  return 1.0;
}

inline double legendre_dpoly_2(const double x) {
  return 3.0 * x;
}

inline double legendre_dpoly_3(const double x) {
  return (7.5 * x * x - 1.5);
}

inline double legendre_dpoly_4(const double x) {
  return (17.5 * x * x * x - 7.5 * x);
}

inline double legendre_dpoly_5(const double x) {
  return (39.375 * x * x * x * x - 26.25 * x * x + 1.875);
}

inline double legendre_dpoly_6(const double x) {
  return (86.625 * x * x * x * x * x - 78.75 * x * x * x + 13.125 * x);
}



#endif  // JAMS_CORE_MATHS_H
