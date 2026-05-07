#ifndef JAMS_TEST_HAMILTONIAN_TEST_CRYSTAL_FIELD_H
#define JAMS_TEST_HAMILTONIAN_TEST_CRYSTAL_FIELD_H

#include "jams/hamiltonian/crystal_field.h"

#include "gtest/gtest.h"

#include <cmath>

namespace {

class CrystalFieldHamiltonianTestAccess : public CrystalFieldHamiltonian {
public:
  using CrystalFieldHamiltonian::convert_spherical_to_tesseral;
};

CrystalFieldHamiltonian::SphericalHarmonicCoefficientMap zero_spherical_coefficients()
{
  CrystalFieldHamiltonian::SphericalHarmonicCoefficientMap coefficients;
  for (auto l : {0, 2, 4, 6}) {
    for (auto m = -l; m <= l; ++m) {
      coefficients.insert({{l, m}, {0.0, 0.0}});
    }
  }
  return coefficients;
}

constexpr int parity_sign(const int m)
{
  return (m % 2 == 0) ? 1 : -1;
}

} // namespace

TEST(CrystalFieldHamiltonianTest, ConvertsPositiveTesseralCoefficients)
{
  constexpr auto inv_sqrt_two = 0.707106781186547524400844362104849039;
  constexpr auto tolerance = 1e-14;

  for (auto l : {2, 4, 6}) {
    for (auto m = 1; m <= l; ++m) {
      auto coefficients = zero_spherical_coefficients();
      const auto phase = static_cast<double>(parity_sign(m));
      coefficients.at({l, -m}) = {inv_sqrt_two, 0.0};
      coefficients.at({l,  m}) = {phase * inv_sqrt_two, 0.0};

      const auto tesseral_coefficients =
          CrystalFieldHamiltonianTestAccess::convert_spherical_to_tesseral(coefficients, tolerance);

      EXPECT_NEAR(tesseral_coefficients.at({l,  m}), 1.0, tolerance) << "l=" << l << " m=" << m;
      EXPECT_NEAR(tesseral_coefficients.at({l, -m}), 0.0, tolerance) << "l=" << l << " m=" << -m;
    }
  }
}

TEST(CrystalFieldHamiltonianTest, ConvertsNegativeTesseralCoefficients)
{
  constexpr auto inv_sqrt_two = 0.707106781186547524400844362104849039;
  constexpr auto tolerance = 1e-14;

  for (auto l : {2, 4, 6}) {
    for (auto m = 1; m <= l; ++m) {
      auto coefficients = zero_spherical_coefficients();
      const auto phase = static_cast<double>(parity_sign(m));
      coefficients.at({l, -m}) = {0.0, inv_sqrt_two};
      coefficients.at({l,  m}) = {0.0, -phase * inv_sqrt_two};

      const auto tesseral_coefficients =
          CrystalFieldHamiltonianTestAccess::convert_spherical_to_tesseral(coefficients, tolerance);

      EXPECT_NEAR(tesseral_coefficients.at({l, -m}), 1.0, tolerance) << "l=" << l << " m=" << -m;
      EXPECT_NEAR(tesseral_coefficients.at({l,  m}), 0.0, tolerance) << "l=" << l << " m=" << m;
    }
  }
}

#endif // JAMS_TEST_HAMILTONIAN_TEST_CRYSTAL_FIELD_H
