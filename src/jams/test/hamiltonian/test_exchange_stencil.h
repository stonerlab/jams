#ifndef JAMS_TEST_HAMILTONIAN_TEST_EXCHANGE_STENCIL_H
#define JAMS_TEST_HAMILTONIAN_TEST_EXCHANGE_STENCIL_H

#include <algorithm>
#include <array>
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
#include "jams/helpers/exception.h"
#include "jams/helpers/output.h"
#include "jams/helpers/utils.h"
#include "jams/test/output.h"

#ifndef JAMS_DIAGNOSTIC_EXCHANGE_REALHI_ACCUMULATION
#define JAMS_DIAGNOSTIC_EXCHANGE_REALHI_ACCUMULATION 0
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
    const std::string& lattice_size = "[4, 3, 2]",
    const std::string& first_backend = "sparse-matrix",
    const std::string& second_module = "exchange",
    const std::string& second_backend = "stencil") {
  const auto backend_setting = [](const std::string& backend) {
    return backend.empty()
        ? std::string{}
        : std::string("        backend = \"") + backend + "\";\n";
  };
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
)CFG" + backend_setting(first_backend) + R"CFG(
)CFG" + extra_exchange_settings + R"CFG(
      },
      {
        module = ")CFG" + second_module + R"CFG(";
        interactions = (
)CFG" + interactions + R"CFG(
        );
        symops = )CFG" + symops + R"CFG(;
)CFG" + backend_setting(second_backend) + R"CFG(
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
#if DO_MIXED_PRECISION
  // Sparse and stencil backends may accumulate equal Real (binary32) terms in
  // different orders.  Do not apply a binary64 tolerance to that comparison.
  const auto effective_tolerance = std::max(tolerance, 5.0e-7);
#else
  const auto effective_tolerance = tolerance;
#endif
  ASSERT_NEAR(expected, actual, effective_tolerance * scale);
}

inline void compare_hamiltonian_outputs(
    Hamiltonian& reference,
    Hamiltonian& candidate,
    const double tolerance) {
  auto spins = make_spins();

  reference.calculate_fields(0.0, spins);
  candidate.calculate_fields(0.0, spins);

  for (auto site = 0; site < globals::num_spins; ++site) {
    for (auto n = 0; n < 3; ++n) {
      assert_near_scaled(reference.field(site, n), candidate.field(site, n), tolerance);
    }
  }

  reference.calculate_energies(0.0, spins);
  candidate.calculate_energies(0.0, spins);
  for (auto site = 0; site < globals::num_spins; ++site) {
    assert_near_scaled(reference.energy(site), candidate.energy(site), tolerance);
  }

  const auto reference_energy = reference.calculate_total_energy(0.0, spins);
  const auto candidate_energy = candidate.calculate_total_energy(0.0, spins);
  assert_near_scaled(reference_energy, candidate_energy, tolerance);

  for (auto site = 0; site < globals::num_spins; ++site) {
    assert_near_scaled(
        reference.calculate_energy(site, 0.0),
        candidate.calculate_energy(site, 0.0),
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
        candidate.calculate_energy_difference(site, spin_initial, spin_final, 0.0),
        tolerance);
  }
}

inline void compare_to_sparse_exchange(
    const libconfig::Setting& settings,
    const double tolerance,
    const bool candidate_is_cuda_solver = false) {
  ExchangeHamiltonian reference(settings[0], globals::num_spins, false);
  ExchangeHamiltonian candidate(settings[1], globals::num_spins, candidate_is_cuda_solver);
  EXPECT_EQ(reference.active_backend(), ExchangeBackend::SparseMatrix);
  EXPECT_EQ(candidate.active_backend(), ExchangeBackend::Stencil);
  compare_hamiltonian_outputs(reference, candidate, tolerance);
}

inline void compare_fields_to_sparse_exchange(
    const libconfig::Setting& settings,
    const double tolerance,
    const bool candidate_is_cuda_solver = false) {
  ExchangeHamiltonian reference(settings[0], globals::num_spins, false);
  ExchangeHamiltonian candidate(settings[1], globals::num_spins, candidate_is_cuda_solver);
  EXPECT_EQ(reference.active_backend(), ExchangeBackend::SparseMatrix);
  EXPECT_EQ(candidate.active_backend(), ExchangeBackend::Stencil);
  auto spins = make_spins();

  reference.calculate_fields(0.0, spins);
  candidate.calculate_fields(0.0, spins);

  double scale = 1.0;
  for (auto site = 0; site < globals::num_spins; ++site) {
    for (auto n = 0; n < 3; ++n) {
      scale = std::max(scale, std::abs(static_cast<double>(reference.field(site, n))));
      scale = std::max(scale, std::abs(static_cast<double>(candidate.field(site, n))));
    }
  }
  for (auto site = 0; site < globals::num_spins; ++site) {
    for (auto n = 0; n < 3; ++n) {
      ASSERT_NEAR(reference.field(site, n), candidate.field(site, n), tolerance * scale);
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
    jams::util::force_deallocation(globals::inv_mus);
    globals::num_magnetic_spins = 0;
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
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForMixedOpenBoundaries) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, false, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false"));
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      tensor_interactions(),
      "false"));
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, MatchesSparseForMultiBasisExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      multi_basis_positions(),
      two_materials(),
      multi_basis_interactions(),
      "false"));
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-8);
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
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, SimpleCubicSixNeighbourFieldErrorIsBounded) {
  using namespace jams::testing::exchange_stencil;
  constexpr int kSize = 16;
  constexpr double kExchangeMeV = 20.0;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      R"(
        ("A", "A", [1.0, 0.0, 0.0], 20.0)
      )",
      "true",
      "",
      R"(
        energy_units = "meV";
      )",
      "",
      "[16, 16, 16]"));

  ExchangeHamiltonian candidate(
      globals::config->lookup("hamiltonians.[1]"), globals::num_spins, false);
  ASSERT_EQ(candidate.active_backend(), ExchangeBackend::Stencil);
  auto spins = make_spins();
  candidate.calculate_fields(0.0, spins);

  const auto wrap = [](const int value) {
    return value < 0 ? value + kSize : (value >= kSize ? value - kSize : value);
  };
  const auto site = [](const int x, const int y, const int z) {
    return (x * kSize + y) * kSize + z;
  };

  double max_abs_error = 0.0;
  double squared_error = 0.0;
  for (auto x = 0; x < kSize; ++x) {
    for (auto y = 0; y < kSize; ++y) {
      for (auto z = 0; z < kSize; ++z) {
        const std::array<int, 6> neighbours = {
            site(wrap(x + 1), y, z),
            site(wrap(x - 1), y, z),
            site(x, wrap(y + 1), z),
            site(x, wrap(y - 1), z),
            site(x, y, wrap(z + 1)),
            site(x, y, wrap(z - 1))};
        const int source = site(x, y, z);
        for (auto component = 0; component < 3; ++component) {
          double expected = 0.0;
          for (const int neighbour : neighbours) {
            expected += kExchangeMeV * static_cast<double>(spins(neighbour, component));
          }
          const double error = static_cast<double>(candidate.field(source, component)) - expected;
          max_abs_error = std::max(max_abs_error, std::abs(error));
          squared_error += error * error;
        }
      }
    }
  }
  const double rms_error = std::sqrt(squared_error / static_cast<double>(globals::num_spins3));

#if DO_MIXED_PRECISION && JAMS_DIAGNOSTIC_EXCHANGE_REALHI_ACCUMULATION
  EXPECT_LE(max_abs_error, 5.0e-6) << "RMS error: " << rms_error;
  EXPECT_LE(rms_error, 1.5e-6);
#elif DO_MIXED_PRECISION
  EXPECT_LE(max_abs_error, 1.1e-5) << "RMS error: " << rms_error;
  EXPECT_LE(rms_error, 2.2e-6);
#else
  EXPECT_LE(max_abs_error, 1.0e-12) << "RMS error: " << rms_error;
  EXPECT_LE(rms_error, 1.0e-13);
#endif
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
        backend = "sparse-matrix";
      },
      {
        module = "exchange";
        exc_file = "exchange_stencil_test_exc.in";
        symops = false;
        backend = "stencil";
      }
    );
  )");
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-8);
  std::remove(file_name.c_str());
}

TEST_F(ExchangeStencilHamiltonianTest, AutoSelectsStencilForDenseExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      "",
      "",
      "[4, 3, 2]",
      "",
      "exchange",
      "sparse-matrix"));
  const auto& settings = globals::config->lookup("hamiltonians");
  ExchangeHamiltonian auto_exchange(settings[0], globals::num_spins, false);
  ExchangeHamiltonian sparse_exchange(settings[1], globals::num_spins, false);
  EXPECT_EQ(auto_exchange.active_backend(), ExchangeBackend::Stencil);
  EXPECT_EQ(sparse_exchange.active_backend(), ExchangeBackend::SparseMatrix);
  compare_hamiltonian_outputs(auto_exchange, sparse_exchange, 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, SelectsSparseMatrixWhenRequested) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      R"(
        tensor_storage = "isotropic";
      )",
      "",
      "[4, 3, 2]",
      "sparse-matrix",
      "exchange",
      "sparse-matrix"));
  const auto& settings = globals::config->lookup("hamiltonians");
  ExchangeHamiltonian reference(settings[0], globals::num_spins, false);
  ExchangeHamiltonian candidate(settings[1], globals::num_spins, false);
  EXPECT_EQ(candidate.active_backend(), ExchangeBackend::SparseMatrix);
  compare_hamiltonian_outputs(reference, candidate, 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, BenchmarkSelectsBackendAndMatchesSparseForDenseExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      "",
      "",
      "[4, 3, 2]",
      "sparse-matrix",
      "exchange",
      "benchmark"));
  const auto& settings = globals::config->lookup("hamiltonians");
  ExchangeHamiltonian reference(settings[0], globals::num_spins, false);
  ExchangeHamiltonian candidate(settings[1], globals::num_spins, false);
  EXPECT_EQ(reference.active_backend(), ExchangeBackend::SparseMatrix);
  EXPECT_TRUE(candidate.active_backend() == ExchangeBackend::Stencil
              || candidate.active_backend() == ExchangeBackend::SparseMatrix);
  compare_hamiltonian_outputs(reference, candidate, 1.0e-8);
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
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, AutoFallsBackToSparseWhenLatticeIsCropped) {
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
      "[4.0, 3.0, 2.5]",
      "sparse-matrix",
      "exchange",
      ""));
  ASSERT_TRUE(globals::lattice->has_cropping());
  const auto& settings = globals::config->lookup("hamiltonians");
  ExchangeHamiltonian reference(settings[0], globals::num_spins, false);
  ExchangeHamiltonian auto_exchange(settings[1], globals::num_spins, false);
  EXPECT_EQ(auto_exchange.active_backend(), ExchangeBackend::SparseMatrix);
  compare_hamiltonian_outputs(reference, auto_exchange, 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, BenchmarkFallsBackToSparseWhenLatticeIsCropped) {
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
      "[4.0, 3.0, 2.5]",
      "sparse-matrix",
      "exchange",
      "benchmark"));
  ASSERT_TRUE(globals::lattice->has_cropping());
  const auto& settings = globals::config->lookup("hamiltonians");
  ExchangeHamiltonian reference(settings[0], globals::num_spins, false);
  ExchangeHamiltonian benchmark_exchange(settings[1], globals::num_spins, false);
  EXPECT_EQ(benchmark_exchange.active_backend(), ExchangeBackend::SparseMatrix);
  compare_hamiltonian_outputs(reference, benchmark_exchange, 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, ForcedStencilRejectsUnsafeCroppedLattice) {
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
  EXPECT_THROW(
      ExchangeHamiltonian(globals::config->lookup("hamiltonians.[1]"), globals::num_spins, false),
      jams::ConfigException);
}

TEST_F(ExchangeStencilHamiltonianTest, BenchmarkAcceptsSparseOnlySettings) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      R"(
        tensor_storage = "isotropic";
        tensor_storage_tolerance = 0.0;
      )",
      "",
      "[4, 3, 2]",
      "sparse-matrix",
      "exchange",
      "benchmark"));
  const auto& settings = globals::config->lookup("hamiltonians");
  ExchangeHamiltonian reference(settings[0], globals::num_spins, false);
  ExchangeHamiltonian candidate(settings[1], globals::num_spins, false);
  EXPECT_TRUE(candidate.active_backend() == ExchangeBackend::Stencil
              || candidate.active_backend() == ExchangeBackend::SparseMatrix);
  compare_hamiltonian_outputs(reference, candidate, 1.0e-8);
}

TEST_F(ExchangeStencilHamiltonianTest, RejectsInvalidExchangeBackend) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      R"(
        backend = "dense";
      )",
      "",
      "[4, 3, 2]",
      "",
      "exchange",
      ""));
  try {
    ExchangeHamiltonian(globals::config->lookup("hamiltonians.[0]"), globals::num_spins, false);
    FAIL() << "expected invalid backend to throw";
  } catch (const jams::ConfigException& error) {
    EXPECT_NE(std::string(error.what()).find("benchmark"), std::string::npos)
        << error.what();
  }
}

TEST_F(ExchangeStencilHamiltonianTest, RejectsSparseOnlySettingsWhenAutoSelectsStencil) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      R"(
        tensor_storage = "isotropic";
      )",
      "",
      "[4, 3, 2]",
      "",
      "exchange",
      "stencil"));
  EXPECT_THROW(
      ExchangeHamiltonian(globals::config->lookup("hamiltonians.[0]"), globals::num_spins, false),
      jams::ConfigException);
}

TEST_F(ExchangeStencilHamiltonianTest, FactoryRejectsLegacyExchangeStencilModule) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      "",
      "",
      "[4, 3, 2]",
      "sparse-matrix",
      "exchange-stencil",
      ""));
  EXPECT_THROW(
      std::unique_ptr<Hamiltonian>(
          Hamiltonian::create(globals::config->lookup("hamiltonians.[1]"), globals::num_spins, false)),
      jams::removed_feature_error);
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
      ExchangeHamiltonian(globals::config->lookup("hamiltonians.[1]"), globals::num_spins, false),
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
      ExchangeHamiltonian(globals::config->lookup("hamiltonians.[1]"), globals::num_spins, false),
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
      ExchangeHamiltonian(globals::config->lookup("hamiltonians.[1]"), globals::num_spins, false),
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
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-5, true);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForMixedOpenBoundaries) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, false, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false"));
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-5, true);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForDiagonalTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      diagonal_tensor_interactions(),
      "false"));
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-5, true);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForSymmetricTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      symmetric_tensor_interactions(),
      "false"));
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-5, true);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForAntisymmetricTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      antisymmetric_tensor_interactions(),
      "false"));
  compare_fields_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-5, true);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaMatchesSparseForGeneralTensorExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      tensor_interactions(),
      "false"));
  compare_to_sparse_exchange(globals::config->lookup("hamiltonians"), 1.0e-5, true);
}

TEST_F(ExchangeStencilHamiltonianTest, CudaBenchmarkSelectsBackendAndMatchesSparseForPeriodicScalarExchange) {
  using namespace jams::testing::exchange_stencil;
  SetUp(make_config(
      "[true, true, true]",
      single_basis_positions(),
      single_material(),
      scalar_interactions(),
      "false",
      "",
      "",
      "",
      "[4, 3, 2]",
      "sparse-matrix",
      "exchange",
      "benchmark"));
  const auto& settings = globals::config->lookup("hamiltonians");
  ExchangeHamiltonian reference(settings[0], globals::num_spins, false);
  ExchangeHamiltonian candidate(settings[1], globals::num_spins, true);
  EXPECT_EQ(reference.active_backend(), ExchangeBackend::SparseMatrix);
  EXPECT_TRUE(candidate.active_backend() == ExchangeBackend::Stencil
              || candidate.active_backend() == ExchangeBackend::SparseMatrix);
  compare_hamiltonian_outputs(reference, candidate, 1.0e-5);
}
#endif  // HAS_CUDA

#endif  // JAMS_TEST_HAMILTONIAN_TEST_EXCHANGE_STENCIL_H
