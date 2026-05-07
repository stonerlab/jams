//
// Created by Joseph Barker on 06/05/2026.
//

#ifndef JAMS_TESSERAL_HARMONICS_H
#define JAMS_TESSERAL_HARMONICS_H
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#ifndef JAMS_HOST_DEVICE
  #ifdef __CUDACC__
    #define JAMS_HOST_DEVICE __host__ __device__
  #else
    #define JAMS_HOST_DEVICE
  #endif
#endif

namespace jams
{
  enum class TesseralHarmonicNormalisation {
    monic,
    condon_shortley,
    racah,
    stevens
  };

  template <typename T>
  JAMS_HOST_DEVICE constexpr T rational(int num, int den = 1) noexcept {

    static_assert(std::is_floating_point<T>::value,
                  "T must be a floating-point type");

    return static_cast<T>(num) / static_cast<T>(den);
  }

  // Z_{2,-2} = x*y
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l2_m2(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*y;
  }

  // Z_{2,-1} = y*z
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l2_m1(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return y*z;
  }

  // Z_{2,0} = -1/3 + z^2
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l2_0(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-1, 3) + z*z;
  }

  // Z_{2,1} = x*z
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l2_p1(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*z;
  }

  // Z_{2,2} = x^2 - y^2
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l2_p2(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*x + T{-1}*y*y;
  }

  // Z_{4,-4} = -1/2*(x*y) + x*y^3 + (x*y*z^2)/2
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_m4(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-1, 2)*x*y + x*y*y*y + rational<T>(1, 2)*x*y*z*z;
  }

  // Z_{4,-3} = x^2*y*z - (y^3*z)/3
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_m3(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*x*y*z + rational<T>(-1, 3)*y*y*y*z;
  }

  // Z_{4,-2} = -1/7*(x*y) + x*y*z^2
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_m2(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-1, 7)*x*y + x*y*z*z;
  }

  // Z_{4,-1} = (-3*y*z)/7 + y*z^3
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_m1(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-3, 7)*y*z + y*z*z*z;
  }

  // Z_{4,0} = 3/35 - (6*z^2)/7 + z^4
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_0(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(3, 35) + rational<T>(-6, 7)*z*z + z*z*z*z;
  }

  // Z_{4,1} = (-3*x*z)/7 + x*z^3
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_p1(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-3, 7)*x*z + x*z*z*z;
  }

  // Z_{4,2} = 1/14 - y^2/7 - (4*z^2)/7 + y^2*z^2 + z^4/2
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_p2(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(1, 14) + rational<T>(-1, 7)*y*y + rational<T>(-4, 7)*z*z + y*y*z*z + rational<T>(1, 2)*z*z*z*z;
  }

  // Z_{4,3} = -1/4*(x*z) + x*y^2*z + (x*z^3)/4
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_p3(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-1, 4)*x*z + x*y*y*z + rational<T>(1, 4)*x*z*z*z;
  }

  // Z_{4,4} = x^4 - 6*x^2*y^2 + y^4
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l4_p4(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*x*x*x + T{-6}*x*x*y*y + y*y*y*y;
  }

  // Z_{6,-6} = x^5*y - (10*x^3*y^3)/3 + x*y^5
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_m6(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*x*x*x*x*y + rational<T>(-10, 3)*x*x*x*y*y*y + x*y*y*y*y*y;
  }

  // Z_{6,-5} = x^4*y*z - 2*x^2*y^3*z + (y^5*z)/5
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_m5(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*x*x*x*y*z + T{-2}*x*x*y*y*y*z + rational<T>(1, 5)*y*y*y*y*y*z;
  }

  // Z_{6,-4} = (x*y)/22 - (x*y^3)/11 - (6*x*y*z^2)/11 + x*y^3*z^2 + (x*y*z^4)/2
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_m4(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(1, 22)*x*y + rational<T>(-1, 11)*x*y*y*y + rational<T>(-6, 11)*x*y*z*z + x*y*y*y*z*z + rational<T>(1, 2)*x*y*z*z*z*z;
  }

  // Z_{6,-3} = (9*y*z)/44 - (3*y^3*z)/11 - (21*y*z^3)/22 + y^3*z^3 + (3*y*z^5)/4
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_m3(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(9, 44)*y*z + rational<T>(-3, 11)*y*y*y*z + rational<T>(-21, 22)*y*z*z*z + y*y*y*z*z*z + rational<T>(3, 4)*y*z*z*z*z*z;
  }

  // Z_{6,-2} = (x*y)/33 - (6*x*y*z^2)/11 + x*y*z^4
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_m2(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(1, 33)*x*y + rational<T>(-6, 11)*x*y*z*z + x*y*z*z*z*z;
  }

  // Z_{6,-1} = (5*y*z)/33 - (10*y*z^3)/11 + y*z^5
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_m1(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(5, 33)*y*z + rational<T>(-10, 11)*y*z*z*z + y*z*z*z*z*z;
  }

  // Z_{6,0} = -5/231 + (5*z^2)/11 - (15*z^4)/11 + z^6
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_0(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-5, 231) + rational<T>(5, 11)*z*z + rational<T>(-15, 11)*z*z*z*z + z*z*z*z*z*z;
  }

  // Z_{6,1} = (5*x*z)/33 - (10*x*z^3)/11 + x*z^5
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_p1(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(5, 33)*x*z + rational<T>(-10, 11)*x*z*z*z + x*z*z*z*z*z;
  }

  // Z_{6,2} = -1/66 + y^2/33 + (19*z^2)/66 - (6*y^2*z^2)/11 - (17*z^4)/22 + y^2*z^4 + z^6/2
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_p2(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-1, 66) + rational<T>(1, 33)*y*y + rational<T>(19, 66)*z*z + rational<T>(-6, 11)*y*y*z*z + rational<T>(-17, 22)*z*z*z*z + y*y*z*z*z*z + rational<T>(1, 2)*z*z*z*z*z*z;
  }

  // Z_{6,3} = (3*x*z)/44 - (3*x*y^2*z)/11 - (7*x*z^3)/22 + x*y^2*z^3 + (x*z^5)/4
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_p3(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(3, 44)*x*z + rational<T>(-3, 11)*x*y*y*z + rational<T>(-7, 22)*x*z*z*z + x*y*y*z*z*z + rational<T>(1, 4)*x*z*z*z*z*z;
  }

  // Z_{6,4} = -1/11*x^4 + (6*x^2*y^2)/11 - y^4/11 + x^4*z^2 - 6*x^2*y^2*z^2 + y^4*z^2
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_p4(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return rational<T>(-1, 11)*x*x*x*x + rational<T>(6, 11)*x*x*y*y + rational<T>(-1, 11)*y*y*y*y + x*x*x*x*z*z + T{-6}*x*x*y*y*z*z + y*y*y*y*z*z;
  }

  // Z_{6,5} = x^5*z - 10*x^3*y^2*z + 5*x*y^4*z
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_p5(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*x*x*x*x*z + T{-10}*x*x*x*y*y*z + T{5}*x*y*y*y*y*z;
  }

  // Z_{6,6} = x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6
  template <typename T>
  JAMS_HOST_DEVICE constexpr T tesseral_monic_polynomial_l6_p6(T x, T y, T z) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    return x*x*x*x*x*x + T{-15}*x*x*x*x*y*y + T{15}*x*x*y*y*y*y + T{-1}*y*y*y*y*y*y;
  }

  // grad Z_{2,-2} = {y, x, 0}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l2_m2_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = y;
    grad[1] = x;
    grad[2] = T{0};
  }

  // grad Z_{2,-1} = {0, z, y}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l2_m1_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = z;
    grad[2] = y;
  }

  // grad Z_{2,0} = {0, 0, 2*z}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l2_0_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = T{0};
    grad[2] = T{2}*z;
  }

  // grad Z_{2,1} = {z, 0, x}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l2_p1_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = z;
    grad[1] = T{0};
    grad[2] = x;
  }

  // grad Z_{2,2} = {2*x, -2*y, 0}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l2_p2_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{2}*x;
    grad[1] = T{-2}*y;
    grad[2] = T{0};
  }

  // grad Z_{4,-4} = {-1/2*y + y^3 + (y*z^2)/2, -1/2*x + 3*x*y^2 + (x*z^2)/2, x*y*z}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_m4_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(-1, 2)*y + y*y*y + rational<T>(1, 2)*y*z*z;
    grad[1] = rational<T>(-1, 2)*x + T{3}*x*y*y + rational<T>(1, 2)*x*z*z;
    grad[2] = x*y*z;
  }

  // grad Z_{4,-3} = {2*x*y*z, x^2*z - y^2*z, x^2*y - y^3/3}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_m3_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{2}*x*y*z;
    grad[1] = x*x*z + T{-1}*y*y*z;
    grad[2] = x*x*y + rational<T>(-1, 3)*y*y*y;
  }

  // grad Z_{4,-2} = {-1/7*y + y*z^2, -1/7*x + x*z^2, 2*x*y*z}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_m2_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(-1, 7)*y + y*z*z;
    grad[1] = rational<T>(-1, 7)*x + x*z*z;
    grad[2] = T{2}*x*y*z;
  }

  // grad Z_{4,-1} = {0, (-3*z)/7 + z^3, (-3*y)/7 + 3*y*z^2}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_m1_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = rational<T>(-3, 7)*z + z*z*z;
    grad[2] = rational<T>(-3, 7)*y + T{3}*y*z*z;
  }

  // grad Z_{4,0} = {0, 0, (-12*z)/7 + 4*z^3}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_0_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = T{0};
    grad[2] = rational<T>(-12, 7)*z + T{4}*z*z*z;
  }

  // grad Z_{4,1} = {(-3*z)/7 + z^3, 0, (-3*x)/7 + 3*x*z^2}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_p1_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(-3, 7)*z + z*z*z;
    grad[1] = T{0};
    grad[2] = rational<T>(-3, 7)*x + T{3}*x*z*z;
  }

  // grad Z_{4,2} = {0, (-2*y)/7 + 2*y*z^2, (-8*z)/7 + 2*y^2*z + 2*z^3}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_p2_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = rational<T>(-2, 7)*y + T{2}*y*z*z;
    grad[2] = rational<T>(-8, 7)*z + T{2}*y*y*z + T{2}*z*z*z;
  }

  // grad Z_{4,3} = {-1/4*z + y^2*z + z^3/4, 2*x*y*z, -1/4*x + x*y^2 + (3*x*z^2)/4}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_p3_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(-1, 4)*z + y*y*z + rational<T>(1, 4)*z*z*z;
    grad[1] = T{2}*x*y*z;
    grad[2] = rational<T>(-1, 4)*x + x*y*y + rational<T>(3, 4)*x*z*z;
  }

  // grad Z_{4,4} = {4*x^3 - 12*x*y^2, -12*x^2*y + 4*y^3, 0}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l4_p4_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{4}*x*x*x + T{-12}*x*y*y;
    grad[1] = T{-12}*x*x*y + T{4}*y*y*y;
    grad[2] = T{0};
  }

  // grad Z_{6,-6} = {5*x^4*y - 10*x^2*y^3 + y^5, x^5 - 10*x^3*y^2 + 5*x*y^4, 0}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_m6_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{5}*x*x*x*x*y + T{-10}*x*x*y*y*y + y*y*y*y*y;
    grad[1] = x*x*x*x*x + T{-10}*x*x*x*y*y + T{5}*x*y*y*y*y;
    grad[2] = T{0};
  }

  // grad Z_{6,-5} = {4*x^3*y*z - 4*x*y^3*z, x^4*z - 6*x^2*y^2*z + y^4*z, x^4*y - 2*x^2*y^3 + y^5/5}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_m5_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{4}*x*x*x*y*z + T{-4}*x*y*y*y*z;
    grad[1] = x*x*x*x*z + T{-6}*x*x*y*y*z + y*y*y*y*z;
    grad[2] = x*x*x*x*y + T{-2}*x*x*y*y*y + rational<T>(1, 5)*y*y*y*y*y;
  }

  // grad Z_{6,-4} = {y/22 - y^3/11 - (6*y*z^2)/11 + y^3*z^2 + (y*z^4)/2, x/22 - (3*x*y^2)/11 - (6*x*z^2)/11 + 3*x*y^2*z^2 + (x*z^4)/2, (-12*x*y*z)/11 + 2*x*y^3*z + 2*x*y*z^3}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_m4_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(1, 22)*y + rational<T>(-1, 11)*y*y*y + rational<T>(-6, 11)*y*z*z + y*y*y*z*z + rational<T>(1, 2)*y*z*z*z*z;
    grad[1] = rational<T>(1, 22)*x + rational<T>(-3, 11)*x*y*y + rational<T>(-6, 11)*x*z*z + T{3}*x*y*y*z*z + rational<T>(1, 2)*x*z*z*z*z;
    grad[2] = rational<T>(-12, 11)*x*y*z + T{2}*x*y*y*y*z + T{2}*x*y*z*z*z;
  }

  // grad Z_{6,-3} = {0, (9*z)/44 - (9*y^2*z)/11 - (21*z^3)/22 + 3*y^2*z^3 + (3*z^5)/4, (9*y)/44 - (3*y^3)/11 - (63*y*z^2)/22 + 3*y^3*z^2 + (15*y*z^4)/4}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_m3_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = rational<T>(9, 44)*z + rational<T>(-9, 11)*y*y*z + rational<T>(-21, 22)*z*z*z + T{3}*y*y*z*z*z + rational<T>(3, 4)*z*z*z*z*z;
    grad[2] = rational<T>(9, 44)*y + rational<T>(-3, 11)*y*y*y + rational<T>(-63, 22)*y*z*z + T{3}*y*y*y*z*z + rational<T>(15, 4)*y*z*z*z*z;
  }

  // grad Z_{6,-2} = {y/33 - (6*y*z^2)/11 + y*z^4, x/33 - (6*x*z^2)/11 + x*z^4, (-12*x*y*z)/11 + 4*x*y*z^3}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_m2_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(1, 33)*y + rational<T>(-6, 11)*y*z*z + y*z*z*z*z;
    grad[1] = rational<T>(1, 33)*x + rational<T>(-6, 11)*x*z*z + x*z*z*z*z;
    grad[2] = rational<T>(-12, 11)*x*y*z + T{4}*x*y*z*z*z;
  }

  // grad Z_{6,-1} = {0, (5*z)/33 - (10*z^3)/11 + z^5, (5*y)/33 - (30*y*z^2)/11 + 5*y*z^4}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_m1_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = rational<T>(5, 33)*z + rational<T>(-10, 11)*z*z*z + z*z*z*z*z;
    grad[2] = rational<T>(5, 33)*y + rational<T>(-30, 11)*y*z*z + T{5}*y*z*z*z*z;
  }

  // grad Z_{6,0} = {0, 0, (10*z)/11 - (60*z^3)/11 + 6*z^5}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_0_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = T{0};
    grad[2] = rational<T>(10, 11)*z + rational<T>(-60, 11)*z*z*z + T{6}*z*z*z*z*z;
  }

  // grad Z_{6,1} = {(5*z)/33 - (10*z^3)/11 + z^5, 0, (5*x)/33 - (30*x*z^2)/11 + 5*x*z^4}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_p1_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(5, 33)*z + rational<T>(-10, 11)*z*z*z + z*z*z*z*z;
    grad[1] = T{0};
    grad[2] = rational<T>(5, 33)*x + rational<T>(-30, 11)*x*z*z + T{5}*x*z*z*z*z;
  }

  // grad Z_{6,2} = {0, (2*y)/33 - (12*y*z^2)/11 + 2*y*z^4, (19*z)/33 - (12*y^2*z)/11 - (34*z^3)/11 + 4*y^2*z^3 + 3*z^5}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_p2_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{0};
    grad[1] = rational<T>(2, 33)*y + rational<T>(-12, 11)*y*z*z + T{2}*y*z*z*z*z;
    grad[2] = rational<T>(19, 33)*z + rational<T>(-12, 11)*y*y*z + rational<T>(-34, 11)*z*z*z + T{4}*y*y*z*z*z + T{3}*z*z*z*z*z;
  }

  // grad Z_{6,3} = {(3*z)/44 - (3*y^2*z)/11 - (7*z^3)/22 + y^2*z^3 + z^5/4, (-6*x*y*z)/11 + 2*x*y*z^3, (3*x)/44 - (3*x*y^2)/11 - (21*x*z^2)/22 + 3*x*y^2*z^2 + (5*x*z^4)/4}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_p3_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(3, 44)*z + rational<T>(-3, 11)*y*y*z + rational<T>(-7, 22)*z*z*z + y*y*z*z*z + rational<T>(1, 4)*z*z*z*z*z;
    grad[1] = rational<T>(-6, 11)*x*y*z + T{2}*x*y*z*z*z;
    grad[2] = rational<T>(3, 44)*x + rational<T>(-3, 11)*x*y*y + rational<T>(-21, 22)*x*z*z + T{3}*x*y*y*z*z + rational<T>(5, 4)*x*z*z*z*z;
  }

  // grad Z_{6,4} = {(-4*x^3)/11 + (12*x*y^2)/11 + 4*x^3*z^2 - 12*x*y^2*z^2, (12*x^2*y)/11 - (4*y^3)/11 - 12*x^2*y*z^2 + 4*y^3*z^2, 2*x^4*z - 12*x^2*y^2*z + 2*y^4*z}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_p4_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = rational<T>(-4, 11)*x*x*x + rational<T>(12, 11)*x*y*y + T{4}*x*x*x*z*z + T{-12}*x*y*y*z*z;
    grad[1] = rational<T>(12, 11)*x*x*y + rational<T>(-4, 11)*y*y*y + T{-12}*x*x*y*z*z + T{4}*y*y*y*z*z;
    grad[2] = T{2}*x*x*x*x*z + T{-12}*x*x*y*y*z + T{2}*y*y*y*y*z;
  }

  // grad Z_{6,5} = {5*x^4*z - 30*x^2*y^2*z + 5*y^4*z, -20*x^3*y*z + 20*x*y^3*z, x^5 - 10*x^3*y^2 + 5*x*y^4}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_p5_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{5}*x*x*x*x*z + T{-30}*x*x*y*y*z + T{5}*y*y*y*y*z;
    grad[1] = T{-20}*x*x*x*y*z + T{20}*x*y*y*y*z;
    grad[2] = x*x*x*x*x + T{-10}*x*x*x*y*y + T{5}*x*y*y*y*y;
  }

  // grad Z_{6,6} = {6*x^5 - 60*x^3*y^2 + 30*x*y^4, -30*x^4*y + 60*x^2*y^3 - 6*y^5, 0}
  template <typename T>
  JAMS_HOST_DEVICE constexpr void tesseral_monic_polynomial_l6_p6_grad(T x, T y, T z, T grad[3]) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    grad[0] = T{6}*x*x*x*x*x + T{-60}*x*x*x*y*y + T{30}*x*y*y*y*y;
    grad[1] = T{-30}*x*x*x*x*y + T{60}*x*x*y*y*y + T{-6}*y*y*y*y*y;
    grad[2] = T{0};
  }

  // Hashes l and m into a single integer for fast lookup.
  //
  // Based on the fact we only ever need l=2,4,6 and
  // -l <= m <= l.
  JAMS_HOST_DEVICE constexpr int tesseral_key(const int l, const int m)
  {
    // 16 * l + (m + 8)
    return (l << 4) | (m + 8);
  }

  constexpr bool valid_tesseral_lm(int l, int m) {
    return (l == 2 || l == 4 || l == 6) &&
           (m >= -l && m <= l);
  }

  template <typename T>
  constexpr T pi() noexcept
  {
    return static_cast<T>(3.141592653589793238462643383279502884L);
  }

  template <typename T>
  T sqrt_rational_over_pi(const int numerator, const int denominator)
  {
    return std::sqrt(rational<T>(numerator, denominator) / pi<T>());
  }

  // Returns the coefficient multiplying the monic polynomial basis function to
  // obtain the unit-normalised real tesseral harmonic with JAMS' phase
  // convention.
  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  T tesseral_condon_shortley_normalisation_scale(const int l, const int m)
  {
    switch (tesseral_key(l, m))
    {
    case tesseral_key(2, -2): return sqrt_rational_over_pi<T>(15, 4);
    case tesseral_key(2, -1): return sqrt_rational_over_pi<T>(15, 4);
    case tesseral_key(2,  0): return sqrt_rational_over_pi<T>(45, 16);
    case tesseral_key(2,  1): return sqrt_rational_over_pi<T>(15, 4);
    case tesseral_key(2,  2): return sqrt_rational_over_pi<T>(15, 16);

    case tesseral_key(4, -4): return sqrt_rational_over_pi<T>(315, 4);
    case tesseral_key(4, -3): return sqrt_rational_over_pi<T>(2835, 32);
    case tesseral_key(4, -2): return sqrt_rational_over_pi<T>(2205, 16);
    case tesseral_key(4, -1): return sqrt_rational_over_pi<T>(2205, 32);
    case tesseral_key(4,  0): return sqrt_rational_over_pi<T>(11025, 256);
    case tesseral_key(4,  1): return sqrt_rational_over_pi<T>(2205, 32);
    case tesseral_key(4,  2): return sqrt_rational_over_pi<T>(2205, 16);
    case tesseral_key(4,  3): return sqrt_rational_over_pi<T>(315, 2);
    case tesseral_key(4,  4): return sqrt_rational_over_pi<T>(315, 256);

    case tesseral_key(6, -6): return sqrt_rational_over_pi<T>(27027, 512);
    case tesseral_key(6, -5): return sqrt_rational_over_pi<T>(225225, 512);
    case tesseral_key(6, -4): return sqrt_rational_over_pi<T>(99099, 16);
    case tesseral_key(6, -3): return sqrt_rational_over_pi<T>(165165, 32);
    case tesseral_key(6, -2): return sqrt_rational_over_pi<T>(1486485, 512);
    case tesseral_key(6, -1): return sqrt_rational_over_pi<T>(297297, 256);
    case tesseral_key(6,  0): return sqrt_rational_over_pi<T>(693693, 1024);
    case tesseral_key(6,  1): return sqrt_rational_over_pi<T>(297297, 256);
    case tesseral_key(6,  2): return sqrt_rational_over_pi<T>(1486485, 512);
    case tesseral_key(6,  3): return sqrt_rational_over_pi<T>(165165, 32);
    case tesseral_key(6,  4): return sqrt_rational_over_pi<T>(99099, 1024);
    case tesseral_key(6,  5): return sqrt_rational_over_pi<T>(9009, 512);
    case tesseral_key(6,  6): return sqrt_rational_over_pi<T>(3003, 2048);
    default:
      throw std::invalid_argument("invalid l,m given to tesseral normalisation scale");
    }
  }

  // Returns the coefficient multiplying the monic polynomial basis function to
  // obtain Racah-normalised real tesseral harmonics:
  //   C_lm = sqrt(4*pi/(2*l + 1)) Y_lm.
  //
  // This lookup is host/device compatible so CPU and CUDA crystal-field code
  // can share the same Racah scaling without duplicating tesseral polynomials.
  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  JAMS_HOST_DEVICE constexpr T tesseral_racah_normalisation_scale_lookup(const int l, const int m)
  {
    switch (tesseral_key(l, m))
    {
    case tesseral_key(2, -2): return T{1.7320508075688772};  // sqrt(3)
    case tesseral_key(2, -1): return T{1.7320508075688772};  // sqrt(3)
    case tesseral_key(2,  0): return T{1.5};                 // sqrt(9/4)
    case tesseral_key(2,  1): return T{1.7320508075688772};  // sqrt(3)
    case tesseral_key(2,  2): return T{0.8660254037844386};  // sqrt(3/4)

    case tesseral_key(4, -4): return T{5.9160797830996161};  // sqrt(35)
    case tesseral_key(4, -3): return T{6.2749501990055663};  // sqrt(315/8)
    case tesseral_key(4, -2): return T{7.8262379212492643};  // sqrt(245/4)
    case tesseral_key(4, -1): return T{5.5339859052946636};  // sqrt(245/8)
    case tesseral_key(4,  0): return T{4.375};               // sqrt(1225/64)
    case tesseral_key(4,  1): return T{5.5339859052946636};  // sqrt(245/8)
    case tesseral_key(4,  2): return T{7.8262379212492643};  // sqrt(245/4)
    case tesseral_key(4,  3): return T{8.3666002653407556};  // sqrt(70)
    case tesseral_key(4,  4): return T{0.73950997288745202}; // sqrt(35/64)

    case tesseral_key(6, -6): return T{4.0301597362883772};  // sqrt(2079/128)
    case tesseral_key(6, -5): return T{11.634069043116428};  // sqrt(17325/128)
    case tesseral_key(6, -4): return T{43.654896632565745};  // sqrt(7623/4)
    case tesseral_key(6, -3): return T{39.851286052020953};  // sqrt(12705/8)
    case tesseral_key(6, -2): return T{29.888464539015718};  // sqrt(114345/128)
    case tesseral_key(6, -1): return T{18.903124741692839};  // sqrt(22869/64)
    case tesseral_key(6,  0): return T{14.4375};             // sqrt(53361/256)
    case tesseral_key(6,  1): return T{18.903124741692839};  // sqrt(22869/64)
    case tesseral_key(6,  2): return T{29.888464539015718};  // sqrt(114345/128)
    case tesseral_key(6,  3): return T{39.851286052020953};  // sqrt(12705/8)
    case tesseral_key(6,  4): return T{5.4568620790707181};  // sqrt(7623/256)
    case tesseral_key(6,  5): return T{2.3268138086232857};  // sqrt(693/128)
    case tesseral_key(6,  6): return T{0.67169328938139616}; // sqrt(231/512)

    default:
#ifdef __CUDA_ARCH__
      return T{0};
#else
      throw std::invalid_argument("invalid l,m given to tesseral normalisation scale");
#endif
    }
  }

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  T tesseral_racah_normalisation_scale(const int l, const int m)
  {
    return tesseral_racah_normalisation_scale_lookup<T>(l, m);
  }

  // Returns the coefficient multiplying the monic polynomial basis function to
  // obtain the classical Stevens tesseral polynomial convention. For example,
  // this maps Z_{2,0} = z^2 - 1/3 to 3z^2 - 1.
  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  constexpr T tesseral_stevens_normalisation_scale(const int l, const int m)
  {
    switch (tesseral_key(l, m))
    {
    case tesseral_key(2, -2): return T{2};
    case tesseral_key(2, -1): return T{1};
    case tesseral_key(2,  0): return T{3};
    case tesseral_key(2,  1): return T{1};
    case tesseral_key(2,  2): return T{1};

    case tesseral_key(4, -4): return T{-8};
    case tesseral_key(4, -3): return T{3};
    case tesseral_key(4, -2): return T{14};
    case tesseral_key(4, -1): return T{7};
    case tesseral_key(4,  0): return T{35};
    case tesseral_key(4,  1): return T{7};
    case tesseral_key(4,  2): return T{-14};
    case tesseral_key(4,  3): return T{-4};
    case tesseral_key(4,  4): return T{1};

    case tesseral_key(6, -6): return T{6};
    case tesseral_key(6, -5): return T{5};
    case tesseral_key(6, -4): return T{-88};
    case tesseral_key(6, -3): return T{-44};
    case tesseral_key(6, -2): return T{66};
    case tesseral_key(6, -1): return T{33};
    case tesseral_key(6,  0): return T{231};
    case tesseral_key(6,  1): return T{33};
    case tesseral_key(6,  2): return T{-66};
    case tesseral_key(6,  3): return T{-44};
    case tesseral_key(6,  4): return T{11};
    case tesseral_key(6,  5): return T{1};
    case tesseral_key(6,  6): return T{1};
    default:
      throw std::invalid_argument("invalid l,m given to tesseral normalisation scale");
    }
  }

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  T tesseral_monic_polynomial_normalisation_scale(
      const TesseralHarmonicNormalisation normalisation,
      const int l,
      const int m)
  {
    switch (normalisation)
    {
    case TesseralHarmonicNormalisation::monic:
      return T{1};
    case TesseralHarmonicNormalisation::condon_shortley:
      return tesseral_condon_shortley_normalisation_scale<T>(l, m);
    case TesseralHarmonicNormalisation::racah:
      return tesseral_racah_normalisation_scale<T>(l, m);
    case TesseralHarmonicNormalisation::stevens:
      return tesseral_stevens_normalisation_scale<T>(l, m);
    default:
      throw std::invalid_argument("invalid tesseral harmonic normalisation");
    }
  }

  // Given an lm hash from `tesseral_key`, returns the value of the tesseral
  // harmonic Z_{l,m}(x, y, z) in the monic polynomial basis.
  template <typename T>
  JAMS_HOST_DEVICE T tesseral_monic_polynomial_key_lookup(const int lm_hash, T x, T y, T z)
  {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    switch (lm_hash)
    {
    case tesseral_key(2, -2): return tesseral_monic_polynomial_l2_m2(x, y, z);
    case tesseral_key(2, -1): return tesseral_monic_polynomial_l2_m1(x, y, z);
    case tesseral_key(2,  0): return tesseral_monic_polynomial_l2_0(x, y, z);
    case tesseral_key(2,  1): return tesseral_monic_polynomial_l2_p1(x, y, z);
    case tesseral_key(2,  2): return tesseral_monic_polynomial_l2_p2(x, y, z);

    case tesseral_key(4, -4): return tesseral_monic_polynomial_l4_m4(x, y, z);
    case tesseral_key(4, -3): return tesseral_monic_polynomial_l4_m3(x, y, z);
    case tesseral_key(4, -2): return tesseral_monic_polynomial_l4_m2(x, y, z);
    case tesseral_key(4, -1): return tesseral_monic_polynomial_l4_m1(x, y, z);
    case tesseral_key(4,  0): return tesseral_monic_polynomial_l4_0(x, y, z);
    case tesseral_key(4,  1): return tesseral_monic_polynomial_l4_p1(x, y, z);
    case tesseral_key(4,  2): return tesseral_monic_polynomial_l4_p2(x, y, z);
    case tesseral_key(4,  3): return tesseral_monic_polynomial_l4_p3(x, y, z);
    case tesseral_key(4,  4): return tesseral_monic_polynomial_l4_p4(x, y, z);

    case tesseral_key(6, -6): return tesseral_monic_polynomial_l6_m6(x, y, z);
    case tesseral_key(6, -5): return tesseral_monic_polynomial_l6_m5(x, y, z);
    case tesseral_key(6, -4): return tesseral_monic_polynomial_l6_m4(x, y, z);
    case tesseral_key(6, -3): return tesseral_monic_polynomial_l6_m3(x, y, z);
    case tesseral_key(6, -2): return tesseral_monic_polynomial_l6_m2(x, y, z);
    case tesseral_key(6, -1): return tesseral_monic_polynomial_l6_m1(x, y, z);
    case tesseral_key(6,  0): return tesseral_monic_polynomial_l6_0(x, y, z);
    case tesseral_key(6,  1): return tesseral_monic_polynomial_l6_p1(x, y, z);
    case tesseral_key(6,  2): return tesseral_monic_polynomial_l6_p2(x, y, z);
    case tesseral_key(6,  3): return tesseral_monic_polynomial_l6_p3(x, y, z);
    case tesseral_key(6,  4): return tesseral_monic_polynomial_l6_p4(x, y, z);
    case tesseral_key(6,  5): return tesseral_monic_polynomial_l6_p5(x, y, z);
    case tesseral_key(6,  6): return tesseral_monic_polynomial_l6_p6(x, y, z);
    default:
#ifdef __CUDA_ARCH__
      return T{0};
#else
      throw std::invalid_argument("invalid hash value given to tesseral_lookup_hash");
#endif
    }
  }

  // Given an lm hash from `tesseral_key`, returns the value of the gradient of
  // the tesseral harmonic grad Z_{l,m}(x, y, z) monic polynomial basis.
  template <typename T>
  JAMS_HOST_DEVICE void tesseral_monic_polynomial_grad_key_lookup(const int lm_hash, T x, T y, T z, T grad[3])
  {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
    switch (lm_hash)
    {
    case tesseral_key(2, -2): tesseral_monic_polynomial_l2_m2_grad(x, y, z, grad); return;
    case tesseral_key(2, -1): tesseral_monic_polynomial_l2_m1_grad(x, y, z, grad); return;
    case tesseral_key(2,  0): tesseral_monic_polynomial_l2_0_grad(x, y, z, grad); return;
    case tesseral_key(2,  1): tesseral_monic_polynomial_l2_p1_grad(x, y, z, grad); return;
    case tesseral_key(2,  2): tesseral_monic_polynomial_l2_p2_grad(x, y, z, grad); return;

    case tesseral_key(4, -4): tesseral_monic_polynomial_l4_m4_grad(x, y, z, grad); return;
    case tesseral_key(4, -3): tesseral_monic_polynomial_l4_m3_grad(x, y, z, grad); return;
    case tesseral_key(4, -2): tesseral_monic_polynomial_l4_m2_grad(x, y, z, grad); return;
    case tesseral_key(4, -1): tesseral_monic_polynomial_l4_m1_grad(x, y, z, grad); return;
    case tesseral_key(4,  0): tesseral_monic_polynomial_l4_0_grad(x, y, z, grad); return;
    case tesseral_key(4,  1): tesseral_monic_polynomial_l4_p1_grad(x, y, z, grad); return;
    case tesseral_key(4,  2): tesseral_monic_polynomial_l4_p2_grad(x, y, z, grad); return;
    case tesseral_key(4,  3): tesseral_monic_polynomial_l4_p3_grad(x, y, z, grad); return;
    case tesseral_key(4,  4): tesseral_monic_polynomial_l4_p4_grad(x, y, z, grad); return;

    case tesseral_key(6, -6): tesseral_monic_polynomial_l6_m6_grad(x, y, z, grad); return;
    case tesseral_key(6, -5): tesseral_monic_polynomial_l6_m5_grad(x, y, z, grad); return;
    case tesseral_key(6, -4): tesseral_monic_polynomial_l6_m4_grad(x, y, z, grad); return;
    case tesseral_key(6, -3): tesseral_monic_polynomial_l6_m3_grad(x, y, z, grad); return;
    case tesseral_key(6, -2): tesseral_monic_polynomial_l6_m2_grad(x, y, z, grad); return;
    case tesseral_key(6, -1): tesseral_monic_polynomial_l6_m1_grad(x, y, z, grad); return;
    case tesseral_key(6,  0): tesseral_monic_polynomial_l6_0_grad(x, y, z, grad); return;
    case tesseral_key(6,  1): tesseral_monic_polynomial_l6_p1_grad(x, y, z, grad); return;
    case tesseral_key(6,  2): tesseral_monic_polynomial_l6_p2_grad(x, y, z, grad); return;
    case tesseral_key(6,  3): tesseral_monic_polynomial_l6_p3_grad(x, y, z, grad); return;
    case tesseral_key(6,  4): tesseral_monic_polynomial_l6_p4_grad(x, y, z, grad); return;
    case tesseral_key(6,  5): tesseral_monic_polynomial_l6_p5_grad(x, y, z, grad); return;
    case tesseral_key(6,  6): tesseral_monic_polynomial_l6_p6_grad(x, y, z, grad); return;
    default:
#ifdef __CUDA_ARCH__
      grad[0] = T{0};
      grad[1] = T{0};
      grad[2] = T{0};
      return;
#else
      throw std::invalid_argument("invalid hash value given to tesseral_lookup_hash");
#endif
    }
  }


  // Returns the value of the tesseral harmonic Z_{l,m}(x, y, z) in the unnormalised
  // Condon–Shortley basis.
  template <typename T>
  JAMS_HOST_DEVICE T tesseral_monic_polynomial(const int l, const int m, T x, T y, T z)
  {
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");

    assert(l == 2 || l == 4 || l == 6);
    assert(m >= -l && m <= l);

    return tesseral_monic_polynomial_key_lookup(tesseral_key(l, m), x, y, z);
  }

}


#endif //JAMS_TESSERAL_HARMONICS_H
