//
// Created by Joseph Barker on 2020-03-09.
//

#ifndef JAMS_TEST_MAT3_H
#define JAMS_TEST_MAT3_H
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <cstdint>
#include <type_traits>

#include "jams/containers/mat3.h"

TEST(MatTest, IsContiguousStandardLayoutStorage) {
  static_assert(sizeof(jams::Mat<double, 3, 3>) == 9 * sizeof(double));
  static_assert(alignof(jams::Mat<double, 3, 3>) == alignof(double));
  static_assert(std::is_trivially_copyable_v<jams::Mat<double, 3, 3>>);
  static_assert(std::is_standard_layout_v<jams::Mat<double, 3, 3>>);

  jams::Mat<double, 3, 3> m{1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0};

  EXPECT_EQ(m.data(), &m[0][0]);
  EXPECT_EQ(m.data() + 4, &m[1][1]);
  EXPECT_EQ(m.data() + 8, &m[2][2]);
}

TEST(MatTest, SupportsGenericDimensionArithmetic) {
  const jams::Mat<int, 3, 2> a{
      1, 2,
      3, 4,
      5, 6};
  const jams::Mat<int, 3, 2> b{
      6, 5,
      4, 3,
      2, 1};

  EXPECT_EQ(a + b, (jams::Mat<int, 3, 2>{
      7, 7,
      7, 7,
      7, 7}));
  EXPECT_EQ(2 * a, (jams::Mat<int, 3, 2>{
      2, 4,
      6, 8,
      10, 12}));
  EXPECT_EQ(a * 2, (jams::Mat<int, 3, 2>{
      2, 4,
      6, 8,
      10, 12}));
  EXPECT_EQ((a * jams::Vec<int, 2>{7, 8}), (jams::Vec<int, 3>{23, 53, 83}));

  const jams::Mat<int, 2, 4> c{
      1, 2, 3, 4,
      5, 6, 7, 8};

  EXPECT_EQ(a * c, (jams::Mat<int, 3, 4>{
      11, 14, 17, 20,
      23, 30, 37, 44,
      35, 46, 57, 68}));
}

TEST(MatTest, SupportsCompoundAssignmentAndPromotedMultiplyAccumulateTypes) {
  static_assert(std::is_same_v<decltype(std::declval<jams::Mat<int, 2, 2>&>() += std::declval<jams::Mat<int, 2, 2>>()),
                               jams::Mat<int, 2, 2>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Mat<int, 2, 2>&>() -= std::declval<jams::Mat<int, 2, 2>>()),
                               jams::Mat<int, 2, 2>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Mat<int, 2, 2>&>() *= 2),
                               jams::Mat<int, 2, 2>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Mat<int, 2, 2>&>() /= 2),
                               jams::Mat<int, 2, 2>&>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Mat<std::int8_t, 2, 2>>() * std::declval<jams::Vec<std::int8_t, 2>>()),
                               jams::Vec<int, 2>>);
  static_assert(std::is_same_v<decltype(std::declval<jams::Mat<std::int8_t, 2, 2>>() * std::declval<jams::Mat<std::int8_t, 2, 2>>()),
                               jams::Mat<int, 2, 2>>);

  jams::Mat<int, 2, 2> a{1, 2, 3, 4};
  const jams::Mat<int, 2, 2> b{5, 6, 7, 8};

  a += b;
  EXPECT_EQ(a, (jams::Mat<int, 2, 2>{6, 8, 10, 12}));

  a -= b;
  EXPECT_EQ(a, (jams::Mat<int, 2, 2>{1, 2, 3, 4}));

  a *= 3;
  EXPECT_EQ(a, (jams::Mat<int, 2, 2>{3, 6, 9, 12}));

  a /= 3;
  EXPECT_EQ(a, (jams::Mat<int, 2, 2>{1, 2, 3, 4}));
}

TEST(MatTest, SupportsGenericIdentityAndCast) {
  constexpr auto id = identity<int, 4>();
  static_assert(id[0][0] == 1);
  static_assert(id[0][1] == 0);
  static_assert(id[3][3] == 1);

  const jams::Mat<int, 2, 2> ints{1, 2, 3, 4};
  const auto doubles = matrix_cast<double>(ints);

  static_assert(std::is_same_v<decltype(doubles), const jams::Mat<double, 2, 2>>);
  EXPECT_DOUBLE_EQ(doubles[1][0], 3.0);
}

class Mat3Test : public ::testing::TestWithParam<std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>> {
    // You can implement all the usual fixture class members here.
    // To access the test parameter, call GetParam() from class
    // TestWithParam<T>.
};

MATCHER_P2(Vec3Eq, expected, tolerance, "") {
  for (auto i = 0; i < 3; ++i) {
    if (std::abs(arg[i] - expected[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

TEST_P(Mat3Test, rotation_matrix_between_vectors) {
  using namespace testing;

  const double eps = 1e-12;
  // Call GetParam() here to get the values
  std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>> const& p = GetParam();
  auto R = rotation_matrix_between_vectors(p.first, p.second);
  auto c = jams::normalize(R * p.first);
  EXPECT_THAT(jams::normalize(p.second), Vec3Eq(c, eps));
}

INSTANTIATE_TEST_SUITE_P(SpecialVectors, Mat3Test, ::testing::Values(
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0,  0.0,  0.0}, { 1.0,  0.0,  0.0}),  // identity (pole)
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0,  0.0,  0.0}, {-1.0,  0.0,  0.0}),  // reversal (pole)
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({-1.0,  0.0,  0.0}, {-1.0,  0.0,  0.0}),  // identity (pole)
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({-1.0,  0.0,  0.0}, { 1.0,  0.0,  0.0}),   // reversal (pole)
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 0.0,  1.0,  0.0}, { 0.0,  1.0,  0.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 0.0,  1.0,  0.0}, { 0.0, -1.0,  0.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 0.0, -1.0,  0.0}, { 0.0, -1.0,  0.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 0.0, -1.0,  0.0}, { 0.0,  1.0,  0.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 0.0,  0.0,  1.0}, { 0.0,  0.0,  1.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 0.0,  0.0,  1.0}, { 0.0,  0.0, -1.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 0.0,  0.0, -1.0}, { 0.0,  0.0, -1.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 0.0,  0.0, -1.0}, { 0.0,  0.0,  1.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0,  1.0,  1.0}, { 1.0,  1.0,  1.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({-1.0, -1.0, -1.0}, {-1.0, -1.0, -1.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0,  1.0,  1.0}, {-1.0, -1.0, -1.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0,  1.0,  0.0}, { 1.0,  1.0,  0.0}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0,  1.0,  0.0}, {-1.0, -1.0,  0.0})
));

INSTANTIATE_TEST_SUITE_P(RandomVectors, Mat3Test, ::testing::Values(
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<jams::Vec<double, 3>, jams::Vec<double, 3>>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)})
));


#endif //JAMS_TEST_MAT3_H
