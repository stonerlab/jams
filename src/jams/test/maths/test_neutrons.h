#ifndef JAMS_TEST_MATHS_TEST_NEUTRONS_H
#define JAMS_TEST_MATHS_TEST_NEUTRONS_H

#include "gtest/gtest.h"

#include <cmath>

#include "jams/helpers/consts.h"
#include "jams/helpers/neutron_units.h"

TEST(NeutronHelpers, MagneticPrefactorUsesClassicalElectronRadiusInBarns) {
  constexpr double expected_prefactor_barn = 0.29061158498820916;
  EXPECT_NEAR(jams::magnetic_neutron_prefactor_barn(), expected_prefactor_barn, 1.0e-16);
}

TEST(NeutronHelpers, CrossSectionScaleIncludesPeriodogramLengthAndUnitCells) {
  constexpr double sample_time_ps = 0.25;
  constexpr int periodogram_length = 8;
  constexpr int periodogram_count = 2;
  constexpr int num_cells = 4;

  const double expected_scale = jams::magnetic_neutron_prefactor_barn()
      * (sample_time_ps * static_cast<double>(periodogram_length))
      / (kTwoPi * kHBarIU * static_cast<double>(periodogram_count) * static_cast<double>(num_cells));

  EXPECT_NEAR(
      jams::neutron_cross_section_barn_mev_scale(
          sample_time_ps,
          periodogram_length,
          periodogram_count,
          num_cells),
      expected_scale,
      1.0e-18);
}

TEST(NeutronHelpers, FormFactorUsesReciprocalLatticeCoordinatesWithoutTwoPi) {
  jams::FormFactorG g = {{0, 2.0}, {2, 0.0}, {4, 0.0}, {6, 0.0}};
  jams::FormFactorJ j = {
      {0, {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      {2, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      {4, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      {6, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};

  EXPECT_NEAR(jams::form_factor({0.0, 0.0, 0.0}, 1.0, g, j), 1.0, 1.0e-15);
  EXPECT_NEAR(jams::form_factor({1.0, 0.0, 0.0}, 1.0, g, j), std::exp(-0.25), 1.0e-15);
}

#endif  // JAMS_TEST_MATHS_TEST_NEUTRONS_H
