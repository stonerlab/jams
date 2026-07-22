#ifndef JAMS_TEST_MATHS_TEST_ANGLES_H
#define JAMS_TEST_MATHS_TEST_ANGLES_H

#include <gtest/gtest.h>

#include "jams/helpers/maths.h"

TEST(PeriodicApproximateEqualityTest, TreatsBranchCutEndpointsAsEquivalent) {
  EXPECT_TRUE(approximately_equal_periodic(180.0, -180.0, 360.0, 1e-8));
  EXPECT_TRUE(approximately_equal_periodic(-180.0, 180.0, 360.0, 1e-8));
}

TEST(PeriodicApproximateEqualityTest, AcceptsRoundoffAcrossBranchCut) {
  EXPECT_TRUE(approximately_equal_periodic(180.0, -179.9999999964, 360.0, 1e-8));
}

TEST(PeriodicApproximateEqualityTest, RejectsRealMismatchAcrossBranchCut) {
  EXPECT_FALSE(approximately_equal_periodic(179.0, -179.0, 360.0, 1e-8));
}

TEST(PeriodicApproximateEqualityTest, RetainsOrdinaryApproximateEqualityAwayFromBranchCut) {
  EXPECT_TRUE(approximately_equal_periodic(90.0, 90.0000005, 360.0, 1e-8));
  EXPECT_FALSE(approximately_equal_periodic(90.0, 90.000005, 360.0, 1e-8));
}

#endif  // JAMS_TEST_MATHS_TEST_ANGLES_H
