#ifndef JAMS_TEST_VEC3_H
#define JAMS_TEST_VEC3_H

#include "gtest/gtest.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <type_traits>

#include "jams/containers/vec3.h"

namespace {

struct CrossOperand {};
struct CrossProductTerm {};
struct CrossComponent {};
struct DotSum {};

constexpr CrossProductTerm operator*(CrossOperand, CrossOperand) {
  return {};
}

constexpr CrossComponent operator-(CrossProductTerm, CrossProductTerm) {
  return {};
}

constexpr DotSum operator+(CrossProductTerm, CrossProductTerm) {
  return {};
}

constexpr DotSum operator+(DotSum, CrossProductTerm) {
  return {};
}

} // namespace

TEST(VecTest, IsContiguousStandardLayoutStorage) {
  static_assert(sizeof(jams::Vec<double, 3>) == 3 * sizeof(double));
  static_assert(alignof(jams::Vec<double, 3>) == alignof(double));
  static_assert(std::is_trivially_copyable_v<jams::Vec<double, 3>>);
  static_assert(std::is_standard_layout_v<jams::Vec<double, 3>>);

  jams::Vec<double, 3> v{1.0, 2.0, 3.0};

  EXPECT_EQ(v.data(), &v[0]);
  EXPECT_EQ(v.data() + 1, &v[1]);
  EXPECT_EQ(v.data() + 2, &v[2]);
}

TEST(VecTest, SupportsStdArrayCompatibilityAndIteration) {
  jams::Vec<double, 3> from_array{std::array<double, 3>{1.0, 2.0, 3.0}};
  std::array<double, 3>& as_array = from_array;
  const std::array<double, 3>& as_const_array = from_array;

  EXPECT_EQ(as_array[1], 2.0);
  EXPECT_EQ(as_const_array[2], 3.0);
  EXPECT_EQ(std::accumulate(from_array.begin(), from_array.end(), 0.0), 6.0);
}

TEST(VecTest, SupportsGenericDimensionArithmetic) {
  const jams::Vec<int, 4> a{1, 2, 3, 4};
  const jams::Vec<int, 4> b{5, 6, 7, 8};

  EXPECT_EQ(a + b, (jams::Vec<int, 4>{6, 8, 10, 12}));
  EXPECT_EQ(2 * a, (jams::Vec<int, 4>{2, 4, 6, 8}));
  EXPECT_EQ(jams::hadamard_product(a, b), (jams::Vec<int, 4>{5, 12, 21, 32}));
  EXPECT_EQ(jams::dot(a, b), 70);
  EXPECT_EQ(jams::sum(a), 10);
  EXPECT_EQ(jams::product(a), 24);
}

TEST(Vec3Test, AngleUsesBothVectorNorms) {
  const jams::Vec<double, 3> a{2.0, 0.0, 0.0};
  const jams::Vec<double, 3> b{2.0, 2.0, 0.0};

  EXPECT_NEAR(jams::angle(a, b), M_PI / 4.0, 1e-14);
}

TEST(Vec3Test, AngleReturnsFloatingTypeAndHandlesZeroVectors) {
  static_assert(std::is_same_v<decltype(jams::angle(std::declval<jams::Vec<int, 3>>(), std::declval<jams::Vec<int, 3>>())), double>);

  EXPECT_TRUE(std::isnan(jams::angle(jams::Vec<double, 3>{0.0, 0.0, 0.0}, jams::Vec<double, 3>{1.0, 0.0, 0.0})));
  EXPECT_DOUBLE_EQ(jams::angle(jams::Vec<double, 3>{1.0, 0.0, 0.0}, jams::Vec<double, 3>{1.0, 0.0, 0.0}), 0.0);
}

TEST(Vec3Test, PolarAngleHandlesZeroVectorAndClampsAxisVectors) {
  EXPECT_DOUBLE_EQ(jams::polar_angle(jams::Vec<double, 3>{0.0, 0.0, 0.0}), 0.0);
  EXPECT_DOUBLE_EQ(jams::polar_angle(jams::Vec<double, 3>{0.0, 0.0, 1.0}), 0.0);
  EXPECT_DOUBLE_EQ(jams::polar_angle(jams::Vec<double, 3>{0.0, 0.0, -1.0}), M_PI);
}

TEST(Vec3Test, UnitVectorDefaultEpsilonIsTypeAware) {
  EXPECT_EQ(jams::unit_vector(jams::Vec<double, 3>{0.0, 0.0, 0.0}), (jams::Vec<double, 3>{0.0, 0.0, 0.0}));
  EXPECT_EQ(jams::unit_vector(jams::Vec<float, 3>{0.0f, 0.0f, 0.0f}), (jams::Vec<float, 3>{0.0f, 0.0f, 0.0f}));
}

TEST(Vec3Test, AbsoluteMaxReturnsAbsoluteValueType) {
  const std::array<std::complex<double>, 2> values{
      std::complex<double>{3.0, 4.0},
      std::complex<double>{1.0, 0.0}};

  static_assert(std::is_same_v<decltype(jams::absolute_max(values)), double>);
  EXPECT_DOUBLE_EQ(jams::absolute_max(values), 5.0);
}

TEST(Vec3Test, NormalizeComponentsUsesComponentMagnitude) {
  EXPECT_EQ(jams::normalize_components(jams::Vec<double, 3>{-2.0, 0.0, 4.0}), (jams::Vec<double, 3>{-1.0, 0.0, 1.0}));
  EXPECT_EQ(jams::normalize_components(jams::Vec<int, 3>{-2, 0, 4}), (jams::Vec<int, 3>{-1, 0, 1}));
}

TEST(Vec3Test, CompoundAssignmentReturnsMutatedVectorReference) {
  static_assert(std::is_same_v<decltype(std::declval<jams::Vec<double, 3>&>() += 1.0), jams::Vec<double, 3>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Vec<double, 3>&>() += std::declval<jams::Vec<double, 3>>()), jams::Vec<double, 3>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Vec<double, 3>&>() -= 1.0), jams::Vec<double, 3>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Vec<double, 3>&>() -= std::declval<jams::Vec<double, 3>>()), jams::Vec<double, 3>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Vec<double, 3>&>() *= 2.0), jams::Vec<double, 3>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Vec<double, 3>&>() /= 2.0), jams::Vec<double, 3>&>);

  jams::Vec<double, 3> a{1.0, 2.0, 3.0};
  jams::Vec<double, 3> b{4.0, 5.0, 6.0};

  EXPECT_EQ(&(a += b), &a);
  EXPECT_EQ((a), (jams::Vec<double, 3>{5.0, 7.0, 9.0}));

  EXPECT_EQ(&(a -= 1.0), &a);
  EXPECT_EQ((a), (jams::Vec<double, 3>{4.0, 6.0, 8.0}));

  EXPECT_EQ(&(a *= 0.5), &a);
  EXPECT_EQ((a), (jams::Vec<double, 3>{2.0, 3.0, 4.0}));

  EXPECT_EQ(&(a /= 2.0), &a);
  EXPECT_EQ((a), (jams::Vec<double, 3>{1.0, 1.5, 2.0}));
}

TEST(Vec3Test, NormSquaredPreservesPromotedArithmeticType) {
  static_assert(std::is_same_v<decltype(jams::norm_squared(std::declval<jams::Vec<std::int8_t, 3>>())), int>);

  const jams::Vec<std::int8_t, 3> a{100, 0, 0};

  EXPECT_EQ(jams::norm_squared(a), 10000);
}

TEST(Vec3Test, CrossReturnTypeUsesFullComponentExpression) {
  static_assert(std::is_same_v<
      decltype(jams::cross(std::declval<jams::Vec<CrossOperand, 3>>(), std::declval<jams::Vec<CrossOperand, 3>>())),
      jams::Vec<CrossComponent, 3>>);
}

TEST(Vec3Test, DotReturnTypeUsesFullSumExpression) {
  static_assert(std::is_same_v<
      decltype(jams::dot(std::declval<jams::Vec<CrossOperand, 3>>(), std::declval<jams::Vec<CrossOperand, 3>>())),
      DotSum>);
}

TEST(Vec3Test, DotSquaredUsesSquaredDotProductType) {
  const jams::Vec<std::int8_t, 3> a{10, 0, 0};

  static_assert(std::is_same_v<decltype(jams::dot_squared(a, a)), int>);
  EXPECT_EQ(jams::dot_squared(a, a), 10000);
}

TEST(Vec3Test, SumAndProductPreservePromotedArithmeticType) {
  const jams::Vec<std::int8_t, 3> sum_values{100, 30, 1};
  const jams::Vec<std::int8_t, 3> product_values{10, 20, 3};

  static_assert(std::is_same_v<decltype(jams::sum(sum_values)), int>);
  static_assert(std::is_same_v<decltype(jams::product(product_values)), int>);
  EXPECT_EQ(jams::sum(sum_values), 131);
  EXPECT_EQ(jams::product(product_values), 600);
}

#endif // JAMS_TEST_VEC3_H
