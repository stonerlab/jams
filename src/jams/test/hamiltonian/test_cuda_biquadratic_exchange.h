#ifndef JAMS_TEST_HAMILTONIAN_TEST_CUDA_BIQUADRATIC_EXCHANGE_H
#define JAMS_TEST_HAMILTONIAN_TEST_CUDA_BIQUADRATIC_EXCHANGE_H

#include <memory>
#include <type_traits>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <libconfig.h++>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"

namespace {

class CudaBiquadraticExchangeTest : public ::testing::Test {
protected:
  using Spin = jams::Vec<double, 3>;
  static constexpr Spin initial_spin{0.6, 0.8, 0.0};
  static constexpr double energy_tolerance = std::is_same_v<jams::Real, float> ? 2e-6 : 1e-12;

  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device is not available";
    }
    jams::Jams::reset_cuda_device();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(R"(
      materials = ({ name = "A"; moment = 2.0; });
      unitcell = {
        parameter = 0.3e-9;
        basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]));
      };
      lattice = {
        size = [2, 1, 1];
        periodic = [false, false, false];
      };
      hamiltonian = {
        module = "biquadratic-exchange";
        energy_units = "meV";
        symops = false;
        interactions = (
          ("A", "A", [ 1.0, 0.0, 0.0], 1.5),
          ("A", "A", [-1.0, 0.0, 0.0], 1.5),
          ("A", "A", [ 0.0, 0.0, 0.0], 0.25)
        );
      };
    )");
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
    ASSERT_EQ(globals::num_spins, 2);
    hamiltonian_.reset(Hamiltonian::create(globals::config->lookup("hamiltonian"),
                                          globals::num_spins, false));
    field_spins_.resize(globals::num_spins, 3);
    set_spin(0, initial_spin);
    set_spin(1, {0.0, 0.6, 0.8});
  }

  void TearDown() override {
    hamiltonian_.reset();
    globals::num_spins = 0;
    globals::num_spins3 = 0;
    globals::num_magnetic_spins = 0;
    jams::util::force_deallocation(globals::s);
    jams::util::force_deallocation(globals::h);
    jams::util::force_deallocation(globals::ds_dt);
    jams::util::force_deallocation(globals::positions);
    jams::util::force_deallocation(globals::alpha);
    jams::util::force_deallocation(globals::mus);
    jams::util::force_deallocation(globals::inv_mus);
    jams::util::force_deallocation(globals::gyro);
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config.reset();
  }

  void set_spin(int site, const Spin& spin) {
    for (int n = 0; n < 3; ++n) {
      globals::s(site, n) = spin[n];
      field_spins_(site, n) = static_cast<jams::Real>(spin[n]);
    }
  }

  double total_energy() const {
    return hamiltonian_->calculate_total_energy(0.0, field_spins_);
  }

  std::unique_ptr<Hamiltonian> hamiltonian_;
  Hamiltonian::SpinArray field_spins_;
};

TEST_F(CudaBiquadraticExchangeTest, TrialChangesMatchTotalEnergy) {
  const auto energy_before = total_energy();
  EXPECT_NEAR(energy_before, -0.25 - 1.5 * 0.48 * 0.48, energy_tolerance);
  EXPECT_NEAR(energy_before,
              hamiltonian_->calculate_energy(0, 0.0) + hamiltonian_->calculate_energy(1, 0.0),
              energy_tolerance);

  const Spin trial_spins[] = {-initial_spin, {0.0, 0.0, 1.0}, 0.5 * initial_spin};
  for (const auto& trial : trial_spins) {
    const auto trial_delta = hamiltonian_->calculate_energy_difference(0, initial_spin, trial, 0.0);
    set_spin(0, trial);
    EXPECT_NEAR(trial_delta, total_energy() - energy_before, energy_tolerance);
    set_spin(0, initial_spin);
  }
}

TEST_F(CudaBiquadraticExchangeTest, FieldsMatchNegativeTotalEnergyGradient) {
  constexpr bool mixed = std::is_same_v<jams::Real, float>;
  constexpr double step = mixed ? 2e-3 : 1e-5;
  constexpr double tolerance = mixed ? 1e-4 : 1e-8;

  hamiltonian_->calculate_fields(0.0, field_spins_);
  ASSERT_EQ(cudaStreamSynchronize(hamiltonian_->get_stream()), cudaSuccess);
  const Spin field{hamiltonian_->field(0, 0), hamiltonian_->field(0, 1),
                   hamiltonian_->field(0, 2)};
  for (int n = 0; n < 3; ++n) {
    auto spin_plus = initial_spin;
    auto spin_minus = initial_spin;
    spin_plus[n] += step;
    spin_minus[n] -= step;
    set_spin(0, spin_plus);
    const auto energy_plus = total_energy();
    set_spin(0, spin_minus);
    const auto energy_minus = total_energy();
    EXPECT_NEAR(field[n], -(energy_plus - energy_minus) / (2.0 * step), tolerance);
  }
}

}  // namespace

#endif
