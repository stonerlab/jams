#pragma once

#include <limits>
#include <memory>

#include <gtest/gtest.h>
#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"

#if HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace jams::testing {

class RandomAnisotropyHamiltonianTest : public ::testing::Test {
protected:
  void SetUp() override {
    config_.readString(R"(
      materials = ({ name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; });
      unitcell = {
        symops = false;
        parameter = 1.0e-9;
        basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]));
      };
      lattice = { size = [2, 1, 1]; periodic = [false, false, false]; };
      hamiltonian = { module = "random-anisotropy"; energy_units = "meV";
                      magnitude = [2.5]; seed = 127; };
    )");
    globals::lattice = &lattice_;
    lattice_.init_from_config(config_);
    globals::s(1, 1) = 1.0;
    globals::s(1, 2) = 0.0;
  }

  void TearDown() override {
    globals::num_spins = 0;
    globals::num_spins3 = 0;
    globals::num_magnetic_spins = 0;
    globals::lattice = nullptr;
    jams::util::force_deallocation(globals::s);
    jams::util::force_deallocation(globals::h);
    jams::util::force_deallocation(globals::ds_dt);
    jams::util::force_deallocation(globals::positions);
    jams::util::force_deallocation(globals::alpha);
    jams::util::force_deallocation(globals::mus);
    jams::util::force_deallocation(globals::inv_mus);
    jams::util::force_deallocation(globals::gyro);
  }

  std::unique_ptr<Hamiltonian> make_hamiltonian(bool use_cuda) {
    return std::unique_ptr<Hamiltonian>(
        Hamiltonian::create(config_.lookup("hamiltonian"), globals::num_spins, use_cuda));
  }

  void check_energy_consistency(
      Hamiltonian& monte_carlo, Hamiltonian& dynamics, [[maybe_unused]] bool use_cuda) {
    Hamiltonian::SpinArray spins(globals::num_spins, 3);
    for (int site = 0; site < globals::num_spins; ++site) {
      for (int component = 0; component < 3; ++component) {
        spins(site, component) = static_cast<jams::Real>(globals::s(site, component));
      }
    }

    constexpr double tolerance = 128 * std::numeric_limits<jams::Real>::epsilon();
    const jams::Vec<double, 3> initial = {0.0, 0.0, 1.0};
    const jams::Vec<double, 3> trial = {0.6, 0.8, 0.0};
    const double initial_energy = dynamics.calculate_total_energy(0.0, spins);
    EXPECT_NEAR(initial_energy, monte_carlo.calculate_total_energy(0.0, spins), tolerance);
    for (int component = 0; component < 3; ++component) {
      spins(0, component) = static_cast<jams::Real>(trial[component]);
    }
    const double final_energy = dynamics.calculate_total_energy(0.0, spins);
    const double trial_delta = monte_carlo.calculate_energy_difference(0, initial, trial, 0.0);
    EXPECT_NEAR(trial_delta, final_energy - initial_energy, tolerance);
    for (int component = 0; component < 3; ++component) {
      EXPECT_DOUBLE_EQ(globals::s(0, component), initial[component]);
      spins(0, component) = static_cast<jams::Real>(initial[component]);
    }

    dynamics.calculate_fields(0.0, spins);
#if HAS_CUDA
    if (use_cuda) {
      ASSERT_EQ(cudaStreamSynchronize(dynamics.get_stream()), cudaSuccess);
    }
#endif
    // The energy is quadratic, so centred differences have no truncation error.
    constexpr jams::Real step = 0.0625;
    for (int site = 0; site < globals::num_spins; ++site) {
      const auto local_field = monte_carlo.calculate_field(site, 0.0);
      for (int component = 0; component < 3; ++component) {
        const auto value = spins(site, component);
        spins(site, component) = value + step;
        const double energy_plus = dynamics.calculate_total_energy(0.0, spins);
        spins(site, component) = value - step;
        const double energy_minus = dynamics.calculate_total_energy(0.0, spins);
        spins(site, component) = value;
        const double negative_gradient = (energy_minus - energy_plus) / (2.0 * step);
        EXPECT_NEAR(dynamics.field(site, component), negative_gradient, tolerance)
            << "site " << site << ", component " << component;
        EXPECT_NEAR(local_field[component], negative_gradient, tolerance)
            << "site " << site << ", component " << component;
      }
    }
  }

private:
  libconfig::Config config_;
  Lattice lattice_;
};

TEST_F(RandomAnisotropyHamiltonianTest, TrialEnergyAndCpuFieldsMatchTotalEnergy) {
  auto hamiltonian = make_hamiltonian(false);
  check_energy_consistency(*hamiltonian, *hamiltonian, false);
}

#if HAS_CUDA
TEST_F(RandomAnisotropyHamiltonianTest, TrialEnergyAndCudaFieldsMatchTotalEnergy) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto cpu_hamiltonian = make_hamiltonian(false);
  auto cuda_hamiltonian = make_hamiltonian(true);
  check_energy_consistency(*cpu_hamiltonian, *cuda_hamiltonian, true);
}
#endif

}  // namespace jams::testing
