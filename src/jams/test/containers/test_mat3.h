//
// Created by Joseph Barker on 2020-03-09.
//

#ifndef JAMS_TEST_MAT3_H
#define JAMS_TEST_MAT3_H
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "jams/containers/mat3.h"

class Mat3Test : public ::testing::TestWithParam<std::pair<Vec3, Vec3>> {
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
  std::pair<Vec3, Vec3> const& p = GetParam();
  auto R = rotation_matrix_between_vectors(p.first, p.second);
  auto c = normalize(R * p.first);
  EXPECT_THAT(normalize(p.second), Vec3Eq(c, eps));
}

INSTANTIATE_TEST_SUITE_P(SpecialVectors, Mat3Test, ::testing::Values(
    std::pair<Vec3, Vec3>({ 1.0,  0.0,  0.0}, { 1.0,  0.0,  0.0}),  // identity (pole)
    std::pair<Vec3, Vec3>({ 1.0,  0.0,  0.0}, {-1.0,  0.0,  0.0}),  // reversal (pole)
    std::pair<Vec3, Vec3>({-1.0,  0.0,  0.0}, {-1.0,  0.0,  0.0}),  // identity (pole)
    std::pair<Vec3, Vec3>({-1.0,  0.0,  0.0}, { 1.0,  0.0,  0.0}),   // reversal (pole)
    std::pair<Vec3, Vec3>({ 0.0,  1.0,  0.0}, { 0.0,  1.0,  0.0}),
    std::pair<Vec3, Vec3>({ 0.0,  1.0,  0.0}, { 0.0, -1.0,  0.0}),
    std::pair<Vec3, Vec3>({ 0.0, -1.0,  0.0}, { 0.0, -1.0,  0.0}),
    std::pair<Vec3, Vec3>({ 0.0, -1.0,  0.0}, { 0.0,  1.0,  0.0}),
    std::pair<Vec3, Vec3>({ 0.0,  0.0,  1.0}, { 0.0,  0.0,  1.0}),
    std::pair<Vec3, Vec3>({ 0.0,  0.0,  1.0}, { 0.0,  0.0, -1.0}),
    std::pair<Vec3, Vec3>({ 0.0,  0.0, -1.0}, { 0.0,  0.0, -1.0}),
    std::pair<Vec3, Vec3>({ 0.0,  0.0, -1.0}, { 0.0,  0.0,  1.0}),
    std::pair<Vec3, Vec3>({ 1.0,  1.0,  1.0}, { 1.0,  1.0,  1.0}),
    std::pair<Vec3, Vec3>({-1.0, -1.0, -1.0}, {-1.0, -1.0, -1.0}),
    std::pair<Vec3, Vec3>({ 1.0,  1.0,  1.0}, {-1.0, -1.0, -1.0}),
    std::pair<Vec3, Vec3>({ 1.0,  1.0,  0.0}, { 1.0,  1.0,  0.0}),
    std::pair<Vec3, Vec3>({ 1.0,  1.0,  0.0}, {-1.0, -1.0,  0.0})
));

INSTANTIATE_TEST_SUITE_P(RandomVectors, Mat3Test, ::testing::Values(
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)}),
    std::pair<Vec3, Vec3>({ 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)},
                          { 1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX),  1.0 - 2.0 * rand()/double(RAND_MAX)})
));


#endif //JAMS_TEST_MAT3_H
