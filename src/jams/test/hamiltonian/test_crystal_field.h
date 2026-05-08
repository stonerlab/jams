#ifndef JAMS_TEST_HAMILTONIAN_TEST_CRYSTAL_FIELD_H
#define JAMS_TEST_HAMILTONIAN_TEST_CRYSTAL_FIELD_H

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/crystal_field.h"
#include "jams/helpers/utils.h"
#include "jams/maths/tesseral_harmonics.h"
#include "jams/test/output.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>

namespace {

class CrystalFieldHamiltonianTestAccess : public CrystalFieldHamiltonian {
public:
  using CrystalFieldHamiltonian::CrystalFieldHamiltonian;
  using CrystalFieldHamiltonian::crystal_field_energy;
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

class CrystalFieldHamiltonianRuntimeTest : public ::testing::Test {
public:
  void SetUp() override
  {
    jams::testing::toggle_cout();

    coefficient_filename_ = "jams_crystal_field_coefficients_test.dat";
    std::ofstream coefficients(coefficient_filename_);
    coefficients << "2 0 1.0 0.0 0.0 0.0\n";
    coefficients.close();

    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(config_string());
    globals::lattice->init_from_config(*globals::config);

    jams::testing::toggle_cout();
  }

  void TearDown() override
  {
    std::remove(coefficient_filename_.c_str());

    globals::num_spins = 0;
    globals::num_spins3 = 0;

    jams::util::force_deallocation(globals::s);
    jams::util::force_deallocation(globals::h);
    jams::util::force_deallocation(globals::ds_dt);
    jams::util::force_deallocation(globals::positions);
    jams::util::force_deallocation(globals::alpha);
    jams::util::force_deallocation(globals::mus);
    jams::util::force_deallocation(globals::gyro);

    globals::config = nullptr;

    if (globals::lattice) {
      delete globals::lattice;
      globals::lattice = nullptr;
    }
  }

protected:
  std::string config_string() const
  {
    return R"(
        solver : {
          module = "llg-heun-gpu";
          t_step = 1.0e-16;
          t_min = 1.0e-16;
          t_max = 1.0e-16;
        };

        materials = (
          { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; }
        );

        unitcell : {
          parameter = 1.0e-9;
          basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
          positions = (("A", [0.0, 0.0, 0.0]));
        };

        lattice : {
          size = [1, 1, 1];
          periodic = [false, false, false];
        };

        hamiltonian = {
          module = "crystal-field";
          energy_units = "meV";
          energy_cutoff = 1e-14;
          crystal_field_spin_type = "up";
          crystal_field_coefficients = (
            ("A", 2.0, 1.0, 0.0, 0.0, ")" + coefficient_filename_ + R"(")
          );
        };
    )";
  }

  std::string coefficient_filename_;
};

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

TEST_F(CrystalFieldHamiltonianRuntimeTest, UsesSparseTesseralKeysForEnergyAndField)
{
  CrystalFieldHamiltonianTestAccess hamiltonian(
      globals::config->lookup("hamiltonian"),
      globals::num_spins);

  constexpr int spin_index = 0;
  const jams::Vec<double, 3> spin = {0.0, 0.0, 1.0};
  const double stevens_prefactor = 2.0 * (2.0 - 0.5);
  const double monic_scale = jams::tesseral_racah_normalisation_scale_lookup<double>(2, 0);
  const double coefficient = stevens_prefactor * monic_scale;

  const double expected_energy = coefficient * jams::tesseral_monic_polynomial(2, 0, spin[0], spin[1], spin[2]);
  ASSERT_NEAR(hamiltonian.crystal_field_energy(spin_index, spin), expected_energy, 1e-14);

  double grad[3];
  jams::tesseral_monic_polynomial_grad_key_lookup(jams::tesseral_key(2, 0), spin[0], spin[1], spin[2], grad);
  const auto field = hamiltonian.calculate_field(spin_index, 0.0);
  for (auto j = 0; j < 3; ++j) {
    ASSERT_NEAR(field[j], -coefficient * grad[j], 1e-14);
  }
}

#endif // JAMS_TEST_HAMILTONIAN_TEST_CRYSTAL_FIELD_H
