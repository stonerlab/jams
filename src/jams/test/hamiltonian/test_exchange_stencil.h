#ifndef JAMS_TEST_HAMILTONIAN_TEST_EXCHANGE_STENCIL_H
#define JAMS_TEST_HAMILTONIAN_TEST_EXCHANGE_STENCIL_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/hamiltonian/exchange_stencil.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/output.h"
#include "jams/helpers/utils.h"
#include "jams/test/output.h"

#if HAS_CUDA
#include "jams/hamiltonian/cuda_exchange_stencil.h"
#endif

namespace jams::testing::exchange_stencil {

inline std::string scalar_interactions() {
  return R"(
        ("A", "A", [ 1.0,  0.0, 0.0],  1.25),
        ("A", "A", [-1.0,  0.0, 0.0],  1.25),
        ("A", "A", [ 0.0,  1.0, 0.0], -0.50),
        ("A", "A", [ 0.0, -1.0, 0.0], -0.50)
  )";
}

inline std::string tensor_interactions() {
  return R"(
        ("A", "A", [ 1.0, 0.0, 0.0], [1.0, 0.2, 0.3, 0.4, 1.5, 0.6, 0.7, 0.8, 2.0]),
        ("A", "A", [-1.0, 0.0, 0.0], [1.0, 0.4, 0.7, 0.2, 1.5, 0.8, 0.3, 0.6, 2.0])
  )";
}

inline std::string diagonal_tensor_interactions() {
  return R"(
        ("A", "A", [ 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0]),
        ("A", "A", [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0])
  )";
}

inline std::string symmetric_tensor_interactions() {
  return R"(
        ("A", "A", [ 1.0, 0.0, 0.0], [1.0, 0.2, 0.3, 0.2, 1.5, 0.6, 0.3, 0.6, 2.0]),
        ("A", "A", [-1.0, 0.0, 0.0], [1.0, 0.2, 0.3, 0.2, 1.5, 0.6, 0.3, 0.6, 2.0])
  )";
}

inline std::string antisymmetric_tensor_interactions() {
  return R"(
        ("A", "A", [ 1.0, 0.0, 0.0], [0.0, 0.2, 0.3, -0.2, 0.0, 0.6, -0.3, -0.6, 0.0]),
        ("A", "A", [-1.0, 0.0, 0.0], [0.0, -0.2, -0.3, 0.2, 0.0, -0.6, 0.3, 0.6, 0.0])
  )";
}

inline std::string multi_basis_interactions() {
  return R"(
        ("A", "B", [ 0.5, 0.0, 0.0],  0.75),
        ("B", "A", [-0.5, 0.0, 0.0],  0.75),
        ("A", "A", [ 0.0, 1.0, 0.0], -0.25),
        ("A", "A", [ 0.0,-1.0, 0.0], -0.25),
        ("B", "B", [ 0.0, 1.0, 0.0],  0.50),
        ("B", "B", [ 0.0,-1.0, 0.0],  0.50)
  )";
}

inline std::string make_config(
    const std::string& periodic,
    const std::string& positions,
    const std::string& materials,
    const std::string& interactions,
    const std::string& symops,
    const std::string& extra_lattice = "",
    const std::string& extra_exchange_settings = "",
    const std::string& extra_root_settings = "",
    const std::string& lattice_size = "[4, 3, 2]") {
  return std::string(R"CFG(
    solver : {
      module = "llg-heun-cpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    unitcell: {
      parameter = 1.0;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
)CFG") + positions + R"CFG(
      );
    };

    materials = (
)CFG" + materials + R"CFG(
    );

    lattice : {
      size = )CFG" + lattice_size + R"CFG(;
      periodic = )CFG" + periodic + R"CFG(;
)CFG" + extra_lattice + R"CFG(
    };

)CFG" + extra_root_settings + R"CFG(

    physics : {
      temperature = 0.0;
    };

    hamiltonians = (
      {
        module = "exchange";
        interactions = (
)CFG" + interactions + R"CFG(
        );
        symops = )CFG" + symops + R"CFG(;
)CFG" + extra_exchange_settings + R"CFG(
      },
      {
        module = "exchange-stencil";
        interactions = (
)CFG" + interactions + R"CFG(
        );
        symops = )CFG" + symops + R"CFG(;
)CFG" + extra_exchange_settings + R"CFG(
      }
    );
  )CFG";
}

inline std::string single_basis_positions() {
  return R"(
        ("A", [0.0, 0.0, 0.0])
  )";
}

inline std::string multi_basis_positions() {
  return R"(
        ("A", [0.0, 0.0, 0.0]),
        ("B", [0.5, 0.0, 0.0])
  )";
}

inline std::string single_material() {
  return R"(
      { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; }
  )";
}

inline std::string two_materials() {
  return R"(
      { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; },
      { name = "B"; moment = 1.0; spin = [0.0, 0.0, 1.0]; }
  )";
}

inline jams::MultiArray<jams::Real, 2> make_spins() {
  jams::MultiArray<jams::Real, 2> spins(globals::num_spins, 3);
  for (auto site = 0; site < globals::num_spins; ++site) {
    const double sx = 0.15 + 0.01 * static_cast<double>(site % 7);
    const double sy = -0.20 + 0.015 * static_cast<double>((site + 2) % 5);
    const double sz = 0.90 - 0.02 * static_cast<double>((site + 3) % 11);
    globals::s(site, 0) = sx;
    globals::s(site, 1) = sy;
    globals::s(site, 2) = sz;
    spins(site, 0) = static_cast<jams::Real>(sx);
    spins(site, 1) = static_cast<jams::Real>(sy);
    spins(site, 2) = static_cast<jams::Real>(sz);
  }
  return spins;
}

inline void assert_near_scaled(const double expected, const double actual, const double tolerance) {
  const auto scale = std::max(1.0, std::max(std::abs(expected), std::abs(actual)));
  ASSERT_NEAR(expected, actual, tolerance * scale);
}

template <typename StencilHamiltonian>
void compare_to_sparse_exchange(const libconfig::Setting& settings, const double tolerance) {
  ExchangeHamiltonian reference(settings[0], globals::num_spins);
  StencilHamiltonian stencil(settings[1], globals::num_spins);
  auto spins = make_spins();

  reference.calculate_fields(0.0, spins);
  stencil.calculate_fields(0.0, spins);

  for (auto site = 0; site < globals::num_spins; ++site) {
    for (auto n = 0; n < 3; ++n) {
      assert_near_scaled(reference.field(site, n), stencil.field(site, n), tolerance);
    }
  }

  reference.calculate_energies(0.0, spins);
  stencil.calculate_energies(0.0, spins);
  for (auto site = 0; site < globals::num_spins; ++site) {
    assert_near_scaled(reference.energy(site), stencil.energy(site), tolerance);
  }

  const auto reference_energy = reference.calculate_total_energy(0.0, spins);
  const auto stencil_energy = stencil.calculate_total_energy(0.0, spins);
  assert_near_scaled(reference_energy, stencil_energy, tolerance);

  for (auto site = 0; site < globals::num_spins; ++site) {
    assert_near_scaled(
        reference.calculate_energy(site, 0.0),
        stencil.calculate_energy(site, 0.0),
        tolerance);
    const jams::Vec<double, 3> spin_initial = {
        globals::s(site, 0),
        globals::s(site, 1),
        globals::s(site, 2)};
    const jams::Vec<double, 3> spin_final = {
        -globals::s(site, 0),
        globals::s(site, 1),
        -globals::s(site, 2)};
    assert_near_scaled(
        reference.calculate_energy_difference(site, spin_initial, spin_final, 0.0),
        stencil.calculate_energy_difference(site, spin_initial, spin_final, 0.0),
        tolerance);
  }
}

template <typename StencilHamiltonian>
void compare_fields_to_sparse_exchange(const libconfig::Setting& settings, const double tolerance) {
  ExchangeHamiltonian reference(settings[0], globals::num_spins);
  StencilHamiltonian stencil(settings[1], globals::num_spins);
  auto spins = make_spins();

  reference.calculate_fields(0.0, spins);
  stencil.calculate_fields(0.0, spins);

  double scale = 1.0;
  for (auto site = 0; site < globals::num_spins; ++site) {
    for (auto n = 0; n < 3; ++n) {
      scale = std::max(scale, std::abs(static_cast<double>(reference.field(site, n))));
      scale = std::max(scale, std::abs(static_cast<double>(stencil.field(site, n))));
    }
  }
  for (auto site = 0; site < globals::num_spins; ++site) {
    for (auto n = 0; n < 3; ++n) {
      ASSERT_NEAR(reference.field(site, n), stencil.field(site, n), tolerance * scale);
    }
  }
}

}  // namespace jams::testing::exchange_stencil

class ExchangeStencilHamiltonianTest : public ::testing::Test {
public:
  void SetUp(const std::string& config_string) {
    jams::testing::toggle_cout();
    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(config_string);
    globals::lattice->init_from_config(*globals::config);
    globals::solver = Solver::create(globals::config->lookup("solver"));
    globals::solver->initialize(globals::config->lookup("solver"));
    globals::solver->register_physics_module(Physics::create(globals::config->lookup("physics")));
    jams::testing::toggle_cout();
  }

  void TearDown() override {
    globals::num_spins = 0;
    globals::num_spins3 = 0;

    jams::util::force_deallocation(globals::s);
    jams::util::force_deallocation(globals::h);
    jams::util::force_deallocation(globals::ds_dt);
    jams::util::force_deallocation(globals::positions);
    jams::util::force_deallocation(globals::alpha);
    jams::util::force_deallocation(globals::mus);
    jams::util::force_deallocation(globals::gyro);

    if (globals::solver) {
      delete globals::solver;
      globals::solver = nullptr;
    }
    globals::config = nullptr;
    if (globals::lattice) {
      delete globals::lattice;
      globals::lattice = nullptr;
    }
  }
};

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForPeriodicScalarExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false"));
  compare_to_sparse_exchange<ExchangeStencilHamiltonian>(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForMixedOpenBoundaries) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, false, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false"));
  compare_to_sparse_exchange<ExchangeStencilHamiltonian>(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      tensor_interactions(),
      "false"));
  compare_to_sparse_exchange<ExchangeStencilHamiltonian>(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForMultiBasisExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      multi_basis_positions(),
      two_materials(),
      multi_basis_interactions(),
      "false"));
  compare_to_sparse_exchange<ExchangeStencilHamiltonian>(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForSymopsGeneratedExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      R"(
        ("A", "A", [0.0, 0.0, 1.0], 0.80)
      )",
      "true",
      "",
      "",
      "",
      "[4, 4, 4]"));
  compare_to_sparse_exchange<ExchangeStencilHamiltonian>(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForExchangeFileInput) {
  using namespace jams::testing::exchange_stencil;
  const std::string file_name = "exchange_stencil_test_exc.in";
  std::remove(file_name.c_str());
  {
    std::ofstream file(file_name);
    file << "A A  1.0 0.0 0.0  1.25\n";
    file << "A A -1.0 0.0 0.0  1.25\n";
    file << "A A  0.0 1.0 0.0 -0.50\n";
    file << "A A  0.0 -1.0 0.0 -0.50\n";
  }

  SetUp(R"(
    solver : {
      module = "llg-heun-cpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };
    unitcell: {
      parameter = 1.0;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0])
      );
    };
    materials = (
      { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; }
    );
    lattice : {
      size = [4, 3, 2];
      periodic = [true, true, true];
    };
    physics : {
      temperature = 0.0;
    };
    hamiltonians = (
      {
        module = "exchange";
        exc_file = "exchange_stencil_test_exc.in";
        symops = false;
      },
      {
        module = "exchange-stencil";
        exc_file = "exchange_stencil_test_exc.in";
        symops = false;
      }
    );
  )");
  compare_to_sparse_exchange<ExchangeStencilHamiltonian>(globals::config->lookup("hamiltonians"), 1.0e-8);
  std::remove(file_name.c_str());
}

TEST_F(ExchangeStencilHamiltonianTest, FallsBackAndMatchesSparseWhenImpuritiesAreConfigured) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      two_materials(),
      scalar_interactions(),
      "false",
      "",
      "",
      R"(
        impurities_seed = 1;
        impurities = (
          ("A", "B", 0.0)
        );
      )"));
  compare_to_sparse_exchange<ExchangeStencilHamiltonian>(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, FallsBackAndMatchesSparseWhenLatticeIsCropped) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, false]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      "",
      "",
      "[4.0, 3.0, 2.5]"));
  ASSERT_TRUE(globals::lattice->has_cropping());
  compare_to_sparse_exchange<ExchangeStencilHamiltonian>(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, RejectsDuplicateStencilTemplates) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      R"(
        ("A", "A", [ 1.0, 0.0, 0.0], 1.0),
        ("A", "A", [ 1.0, 0.0, 0.0], 1.0),
        ("A", "A", [-1.0, 0.0, 0.0], 1.0)
      )",
      "false"));
  EXPECT_THROW(
      ExchangeStencilHamiltonian(globals::config->lookup("hamiltonians.[1]"), globals::num_spins),
      std::runtime_error);
}

TEST_F(ExchangeStencilHamiltonianTest, RejectsDuplicatePhysicalTargetsAfterPeriodicWrapping) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      R"(
        ("A", "A", [ 0.0, 0.0,  1.0], 1.0),
        ("A", "A", [ 0.0, 0.0, -1.0], 1.0)
      )",
      "false",
      "",
      "",
      "",
      "[4, 4, 2]"));
  EXPECT_THROW(
      ExchangeStencilHamiltonian(globals::config->lookup("hamiltonians.[1]"), globals::num_spins),
      std::runtime_error);
}

TEST_F(ExchangeStencilHamiltonianTest, RejectsMissingReverseInteraction) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      R"(
        ("A", "A", [1.0, 0.0, 0.0], 1.0)
      )",
      "false"));
  EXPECT_THROW(
      ExchangeStencilHamiltonian(globals::config->lookup("hamiltonians.[1]"), globals::num_spins),
      jams::SanityException);
}

#if HAS_CUDA
TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForPeriodicScalarExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false"));
  compare_to_sparse_exchange<CudaExchangeStencilHamiltonian>(
      globals::config->lookup("hamiltonians"),
      1.0e-5);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForMixedOpenBoundaries) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, false, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false"));
  compare_to_sparse_exchange<CudaExchangeStencilHamiltonian>(
      globals::config->lookup("hamiltonians"),
      1.0e-5);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForDiagonalTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      diagonal_tensor_interactions(),
      "false"));
  compare_to_sparse_exchange<CudaExchangeStencilHamiltonian>(
      globals::config->lookup("hamiltonians"),
      1.0e-5);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForSymmetricTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      symmetric_tensor_interactions(),
      "false"));
  compare_to_sparse_exchange<CudaExchangeStencilHamiltonian>(
      globals::config->lookup("hamiltonians"),
      1.0e-5);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForAntisymmetricTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      antisymmetric_tensor_interactions(),
      "false"));
  compare_fields_to_sparse_exchange<CudaExchangeStencilHamiltonian>(
      globals::config->lookup("hamiltonians"),
      1.0e-5);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForGeneralTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      tensor_interactions(),
      "false"));
  compare_to_sparse_exchange<CudaExchangeStencilHamiltonian>(
      globals::config->lookup("hamiltonians"),
      1.0e-5);
}
#endif  // HAS_CUDA

#endif  // JAMS_TEST_HAMILTONIAN_TEST_EXCHANGE_STENCIL_H
