//
// Created by Joe Barker on 2017/12/04.
//

#ifndef JAMS_TEST_CELL_H
#define JAMS_TEST_CELL_H
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "jams/containers/cell.h"

//MATCHER_P2(Vec3Eq, expected, tolerance, "") {
//  for (auto i = 0; i < 3; ++i) {
//    if (std::abs(arg[i] - expected[i]) > tolerance) {
//      return false;
//    }
//  }
//  return true;
//}

MATCHER_P2(Mat3Eq, expected, tolerance, "") {
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      if (std::abs(arg[i][j] - expected[i][j]) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

TEST(CellTest, ctor) {
  using namespace testing;

  const double eps = 1e-6;

  Vec3 a = {1.00, 0.00, 0.00};
  Vec3 b = {0.25, 0.50, 0.00};
  Vec3 c = {0.25, 0.25, 0.75};

  Mat3 unitcell = {1.00, 0.25, 0.25, 0.00, 0.50, 0.25, 0.00, 0.00, 0.75};
  Mat3 inverse_matrix = {1.0, -0.5, -0.1666667, 0.0, 2.0, -0.6666667, 0.0, 0.0, 1.333333};


  Cell x(a, b, c);

  ASSERT_THAT(x.a(), Vec3Eq(a, eps));
  ASSERT_THAT(x.b(), Vec3Eq(b, eps));
  ASSERT_THAT(x.c(), Vec3Eq(c, eps));

  ASSERT_THAT(x.matrix(), Mat3Eq(unitcell, eps));
  ASSERT_THAT(x.inverse_matrix(), Mat3Eq(inverse_matrix, eps));
}

TEST(CellTest, minimum_image) {
  using namespace testing;

  const double eps = 1e-6;

  Vec3 a = {10.00, 0.00, 0.00};
  Vec3 b = {0.00, 10.00, 0.00};
  Vec3 c = {0.00, 0.00, 10.00};

  Cell x(a, b, c);

  auto r_ij = minimum_image(x, {0, 0, 0}, {1, 1, 1});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{1, 1, 1}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {1, 1, 1});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{1, 1, 1}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {0, 0, 1});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{0, 0, 1}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {0, 1, 0});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{0, 1, 0}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {1, 0, 0});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{1, 0, 0}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {5, 5, 5});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{-5, -5, -5}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {-5, -5, -5});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{5, 5, 5}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {0, 0, 7});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{0, 0, -3}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {0, 7, 0});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{0, -3, 0}}, eps));

  r_ij = minimum_image(x, {0, 0, 0}, {7, 0, 0});
  ASSERT_THAT(r_ij, Vec3Eq(Vec3{{-3, 0, 0}}, eps));
}

#endif //JAMS_TEST_CELL_H
