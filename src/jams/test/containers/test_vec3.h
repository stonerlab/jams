#ifndef JAMS_TEST_VEC3_H
#define JAMS_TEST_VEC3_H

#include "gtest/gtest.h"

#include <cstdint>
#include <type_traits>

#include "jams/containers/vec3.h"

namespace {

struct CrossOperand {};
struct CrossProductTerm {};
struct CrossComponent {};

constexpr CrossProductTerm operator*(CrossOperand, CrossOperand) {
  return {};
}

constexpr CrossComponent operator-(CrossProductTerm, CrossProductTerm) {
  return {};
}

} // namespace

TEST(Vec3Test, AngleUsesBothVectorNorms) {
  const Vec3 a{2.0, 0.0, 0.0};
  const Vec3 b{2.0, 2.0, 0.0};

  EXPECT_NEAR(jams::angle(a, b), M_PI / 4.0, 1e-14);
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

#endif // JAMS_TEST_VEC3_H
