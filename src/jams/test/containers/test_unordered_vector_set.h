//
// Created by Joseph Barker on 2019-10-16.
//

#ifndef JAMS_TEST_UNORDERED_VECTOR_SET_H
#define JAMS_TEST_UNORDERED_VECTOR_SET_H

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "jams/containers/vec3.h"
#include "jams/containers/mat3.h"

#include "jams/containers/unordered_vector_set.h"

// test array cast
TEST(UnorderedVectorSetTest, insert) {
  using namespace jams;
  using namespace testing;

  Mat3 A = kIdentityMat3;
  Mat3 B = 2.0 * kIdentityMat3;
  Mat3 C = -3.0 * kIdentityMat3;
  Mat3 D = 3.0 * kIdentityMat3;

  UnorderedVectorSet<Mat3> vset;

  ASSERT_EQ(vset.size(), 0);

  vset.insert(A);

  ASSERT_EQ(vset.size(), 1);

  ASSERT_TRUE(vset.insert(B).second);

  ASSERT_EQ(vset.size(), 2);



  ASSERT_FALSE(vset.insert(B).second);

  ASSERT_EQ(vset.size(), 2);

  ASSERT_TRUE(vset.insert(C).second);

  ASSERT_EQ(vset.size(), 3);

  ASSERT_TRUE(vset.insert(D).second);

  ASSERT_EQ(vset.size(), 4);

  ASSERT_THAT(vset[0][0], ElementsAre(1.0, 0.0, 0.0));
  ASSERT_THAT(vset[0][1], ElementsAre(0.0, 1.0, 0.0));
  ASSERT_THAT(vset[0][2], ElementsAre(0.0, 0.0, 1.0));

  ASSERT_THAT(vset[1][0], ElementsAre(2.0, 0.0, 0.0));
  ASSERT_THAT(vset[1][1], ElementsAre(0.0, 2.0, 0.0));
  ASSERT_THAT(vset[1][2], ElementsAre(0.0, 0.0, 2.0));

  ASSERT_THAT(vset[2][0], ElementsAre(-3.0, 0.0, 0.0));
  ASSERT_THAT(vset[2][1], ElementsAre(0.0, -3.0, 0.0));
  ASSERT_THAT(vset[2][2], ElementsAre(0.0, 0.0, -3.0));

  ASSERT_THAT(vset[3][0], ElementsAre(3.0, 0.0, 0.0));
  ASSERT_THAT(vset[3][1], ElementsAre(0.0, 3.0, 0.0));
  ASSERT_THAT(vset[3][2], ElementsAre(0.0, 0.0, 3.0));

}

#endif //JAMS_TEST_UNORDERED_VECTOR_SET_H
