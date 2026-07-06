#ifndef JAMS_TEST_CORE_TEST_MAGNETIC_MOMENTS_H
#define JAMS_TEST_CORE_TEST_MAGNETIC_MOMENTS_H

#include "gtest/gtest.h"

#include <cmath>
#include <limits>

#include "jams/core/globals.h"

namespace {

class MagneticMomentGlobalsTest : public ::testing::Test {
 protected:
  void TearDown() override {
    globals::num_spins = 0;
    globals::num_spins3 = 0;
    globals::num_magnetic_spins = 0;
    globals::s.clear();
    globals::mus.clear();
    globals::inv_mus.clear();
  }
};

TEST_F(MagneticMomentGlobalsTest, SyncBuildsInverseMomentsAndZerosVacancySpin) {
  globals::num_spins = 3;
  globals::num_spins3 = 9;
  globals::mus.resize(3);
  globals::mus(0) = jams::Real{2.0};
  globals::mus(1) = jams::Real{0.0};
  globals::mus(2) = jams::Real{4.0};
  globals::s.resize(3, 3);
  globals::s(0, 0) = 1.0;
  globals::s(0, 1) = 0.0;
  globals::s(0, 2) = 0.0;
  globals::s(1, 0) = 0.0;
  globals::s(1, 1) = 1.0;
  globals::s(1, 2) = 0.0;
  globals::s(2, 0) = 0.0;
  globals::s(2, 1) = 0.0;
  globals::s(2, 2) = 1.0;

  globals::sync_magnetic_moment_data();

  EXPECT_EQ(globals::num_magnetic_spins, 2);
  EXPECT_EQ(globals::inv_mus(0), jams::Real{0.5});
  EXPECT_EQ(globals::inv_mus(1), jams::Real{0.0});
  EXPECT_EQ(globals::inv_mus(2), jams::Real{0.25});
  EXPECT_EQ(globals::s(1, 0), 0.0);
  EXPECT_EQ(globals::s(1, 1), 0.0);
  EXPECT_EQ(globals::s(1, 2), 0.0);
}

TEST_F(MagneticMomentGlobalsTest, SyncRejectsNegativeMoment) {
  globals::num_spins = 1;
  globals::num_spins3 = 3;
  globals::mus.resize(1);
  globals::mus(0) = jams::Real{-1.0};

  EXPECT_THROW(globals::sync_magnetic_moment_data(), std::runtime_error);
}

TEST_F(MagneticMomentGlobalsTest, SyncRejectsNonFiniteMoment) {
  globals::num_spins = 1;
  globals::num_spins3 = 3;
  globals::mus.resize(1);
  globals::mus(0) = static_cast<jams::Real>(std::numeric_limits<double>::quiet_NaN());

  EXPECT_THROW(globals::sync_magnetic_moment_data(), std::runtime_error);
}

TEST_F(MagneticMomentGlobalsTest, FirstMagneticSpinSkipsVacancies) {
  globals::num_spins = 3;
  globals::num_spins3 = 9;
  globals::mus.resize(3);
  globals::mus(0) = jams::Real{0.0};
  globals::mus(1) = jams::Real{0.0};
  globals::mus(2) = jams::Real{3.0};

  globals::sync_magnetic_moment_data();

  EXPECT_EQ(globals::first_magnetic_spin(), 2);
}

}  // namespace

#endif  // JAMS_TEST_CORE_TEST_MAGNETIC_MOMENTS_H
