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
  static_assert(sizeof(Vec3) == 3 * sizeof(double));
  static_assert(alignof(Vec3) == alignof(double));
  static_assert(std::is_trivially_copyable_v<Vec3>);
  static_assert(std::is_standard_layout_v<Vec3>);

  Vec3 v{1.0, 2.0, 3.0};

  EXPECT_EQ(v.data(), &v[0]);
  EXPECT_EQ(v.data() + 1, &v[1]);
  EXPECT_EQ(v.data() + 2, &v[2]);
}

TEST(VecTest, SupportsStdArrayCompatibilityAndIteration) {
  Vec3 from_array{std::array<double, 3>{1.0, 2.0, 3.0}};
  std::array<double, 3>& as_array = from_array;
  const std::array<double, 3>& as_const_array = from_array;

  EXPECT_EQ(as_array[1], 2.0);
  EXPECT_EQ(as_const_array[2], 3.0);
  EXPECT_EQ(std::accumulate(from_array.begin(), from_array.end(), 0.0), 6.0);
}

TEST(Vec3Test, AngleUsesBothVectorNorms) {
  const Vec3 a{2.0, 0.0, 0.0};
  const Vec3 b{2.0, 2.0, 0.0};

  EXPECT_NEAR(jams::angle(a, b), M_PI / 4.0, 1e-14);
}

TEST(Vec3Test, AngleReturnsFloatingTypeAndHandlesZeroVectors) {
  static_assert(std::is_same_v<decltype(jams::angle(std::declval<Vec3i>(), std::declval<Vec3i>())), double>);

  EXPECT_TRUE(std::isnan(jams::angle(Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0})));
  EXPECT_DOUBLE_EQ(jams::angle(Vec3{1.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}), 0.0);
}

TEST(Vec3Test, PolarAngleHandlesZeroVectorAndClampsAxisVectors) {
  EXPECT_DOUBLE_EQ(jams::polar_angle(Vec3{0.0, 0.0, 0.0}), 0.0);
  EXPECT_DOUBLE_EQ(jams::polar_angle(Vec3{0.0, 0.0, 1.0}), 0.0);
  EXPECT_DOUBLE_EQ(jams::polar_angle(Vec3{0.0, 0.0, -1.0}), M_PI);
}

TEST(Vec3Test, UnitVectorDefaultEpsilonIsTypeAware) {
  EXPECT_EQ(jams::unit_vector(Vec3{0.0, 0.0, 0.0}), (Vec3{0.0, 0.0, 0.0}));
  EXPECT_EQ(jams::unit_vector(Vec3f{0.0f, 0.0f, 0.0f}), (Vec3f{0.0f, 0.0f, 0.0f}));
}

TEST(Vec3Test, AbsoluteMaxReturnsAbsoluteValueType) {
  const std::array<std::complex<double>, 2> values{
      std::complex<double>{3.0, 4.0},
      std::complex<double>{1.0, 0.0}};

  static_assert(std::is_same_v<decltype(jams::absolute_max(values)), double>);
  EXPECT_DOUBLE_EQ(jams::absolute_max(values), 5.0);
}

TEST(Vec3Test, NormalizeComponentsUsesComponentMagnitude) {
  EXPECT_EQ(jams::normalize_components(Vec3{-2.0, 0.0, 4.0}), (Vec3{-1.0, 0.0, 1.0}));
  EXPECT_EQ(jams::normalize_components(Vec3i{-2, 0, 4}), (Vec3i{-1, 0, 1}));
}

TEST(Vec3Test, CompoundAssignmentReturnsMutatedVectorReference) {
  static_assert(std::is_same_v<decltype(std::declval<Vec3&>() += 1.0), Vec3&>);
  static_assert(std::is_same_v<decltype(std::declval<Vec3&>() += std::declval<Vec3>()), Vec3&>);
  static_assert(std::is_same_v<decltype(std::declval<Vec3&>() -= 1.0), Vec3&>);
  static_assert(std::is_same_v<decltype(std::declval<Vec3&>() -= std::declval<Vec3>()), Vec3&>);
  static_assert(std::is_same_v<decltype(std::declval<Vec3&>() *= 2.0), Vec3&>);
  static_assert(std::is_same_v<decltype(std::declval<Vec3&>() /= 2.0), Vec3&>);

  Vec3 a{1.0, 2.0, 3.0};
  Vec3 b{4.0, 5.0, 6.0};

  EXPECT_EQ(&(a += b), &a);
  EXPECT_EQ((a), (Vec3{5.0, 7.0, 9.0}));

  EXPECT_EQ(&(a -= 1.0), &a);
  EXPECT_EQ((a), (Vec3{4.0, 6.0, 8.0}));

  EXPECT_EQ(&(a *= 0.5), &a);
  EXPECT_EQ((a), (Vec3{2.0, 3.0, 4.0}));

  EXPECT_EQ(&(a /= 2.0), &a);
  EXPECT_EQ((a), (Vec3{1.0, 1.5, 2.0}));
}

TEST(Vec3Test, NormSquaredPreservesPromotedArithmeticType) {
  static_assert(std::is_same_v<decltype(jams::norm_squared(std::declval<Vec<std::int8_t, 3>>())), int>);

  const Vec<std::int8_t, 3> a{100, 0, 0};

  EXPECT_EQ(jams::norm_squared(a), 10000);
}

TEST(Vec3Test, CrossReturnTypeUsesFullComponentExpression) {
  static_assert(std::is_same_v<
      decltype(jams::cross(std::declval<Vec<CrossOperand, 3>>(), std::declval<Vec<CrossOperand, 3>>())),
      Vec<CrossComponent, 3>>);
}

TEST(Vec3Test, DotReturnTypeUsesFullSumExpression) {
  static_assert(std::is_same_v<
      decltype(jams::dot(std::declval<Vec<CrossOperand, 3>>(), std::declval<Vec<CrossOperand, 3>>())),
      DotSum>);
}

TEST(Vec3Test, DotSquaredUsesSquaredDotProductType) {
  const Vec<std::int8_t, 3> a{10, 0, 0};

  static_assert(std::is_same_v<decltype(jams::dot_squared(a, a)), int>);
  EXPECT_EQ(jams::dot_squared(a, a), 10000);
}

TEST(Vec3Test, SumAndProductPreservePromotedArithmeticType) {
  const Vec<std::int8_t, 3> sum_values{100, 30, 1};
  const Vec<std::int8_t, 3> product_values{10, 20, 3};

  static_assert(std::is_same_v<decltype(jams::sum(sum_values)), int>);
  static_assert(std::is_same_v<decltype(jams::product(product_values)), int>);
  EXPECT_EQ(jams::sum(sum_values), 131);
  EXPECT_EQ(jams::product(product_values), 600);
}

#endif // JAMS_TEST_VEC3_H
