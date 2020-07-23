//
// Created by Joseph Barker on 2019-10-16.
//

#ifndef JAMS_TEST_VECTOR_SET_H
#define JAMS_TEST_VECTOR_SET_H

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "jams/containers/vec3.h"
#include "jams/containers/mat3.h"

#include "jams/containers/vector_set.h"

TEST(VectorSetTest, double) {
using namespace jams;
using namespace testing;

double A = 1.0;
double B = 2.0;
double C = -3.0;

VectorSet<double> vset;

ASSERT_EQ(vset.size(), 0);
ASSERT_TRUE(vset.insert(A).second);
ASSERT_EQ(vset.size(), 1);
ASSERT_TRUE(vset.insert(B).second);
ASSERT_EQ(vset.size(), 2);
ASSERT_FALSE(vset.insert(B).second);
ASSERT_EQ(vset.size(), 2);
ASSERT_TRUE(vset.insert(C).second);
ASSERT_EQ(vset.size(), 3);
ASSERT_EQ(vset[0], -3.0);
ASSERT_EQ(vset[1], 1.0);
ASSERT_EQ(vset[2], 2.0);
}

TEST(VectorSetTest, Vec3) {
  using namespace jams;
  using namespace testing;

  Vec3 A = {1.0, 1.0, 0.0};
  Vec3 B = {1.0, 1.0, -1.0};
  Vec3 C = {-3.0, 3.0, 1.0};

  VectorSet<Vec3> vset;

  ASSERT_EQ(vset.size(), 0);
  ASSERT_TRUE(vset.insert(A).second);
  ASSERT_EQ(vset.size(), 1);
  ASSERT_TRUE(vset.insert(B).second);
  ASSERT_EQ(vset.size(), 2);
  ASSERT_FALSE(vset.insert(B).second);
  ASSERT_EQ(vset.size(), 2);
  ASSERT_TRUE(vset.insert(C).second);
  ASSERT_EQ(vset.size(), 3);
  ASSERT_THAT(vset[0], ElementsAre(-3.0, 3.0, 1.0));
  ASSERT_THAT(vset[1], ElementsAre(1.0, 1.0,-1.0));
  ASSERT_THAT(vset[2], ElementsAre(1.0, 1.0, 0.0));
}

#endif //JAMS_TEST_VECTOR_SET_H
