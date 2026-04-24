#ifndef JAMS_TEST_VEC3_H
#define JAMS_TEST_VEC3_H

#include "gtest/gtest.h"

#include "jams/containers/vec3.h"

TEST(Vec3Test, AngleUsesBothVectorNorms) {
  const Vec3 a{2.0, 0.0, 0.0};
  const Vec3 b{2.0, 2.0, 0.0};

  EXPECT_NEAR(jams::angle(a, b), M_PI / 4.0, 1e-14);
}

#endif // JAMS_TEST_VEC3_H
