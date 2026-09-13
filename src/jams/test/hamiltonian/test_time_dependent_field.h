#ifndef JAMS_TEST_HAMILTONIAN_TEST_TIME_DEPENDENT_FIELD_H
#define JAMS_TEST_HAMILTONIAN_TEST_TIME_DEPENDENT_FIELD_H

#include "jams/hamiltonian/applied_field.h"
#include "jams/hamiltonian/field_pulse.h"
#include "jams/hamiltonian/zeeman.h"
#include "jams/test/monitors/test_magnetisation.h"

#if HAS_CUDA
#include "jams/hamiltonian/cuda_applied_field.h"
#include "jams/hamiltonian/cuda_field_pulse.h"
#endif

namespace jams::testing {

// Reuse the monitor fixture's lattice and temporary-output lifecycle.
class TimeDependentFieldTest : public MagnetisationMonitorTest {
 protected:
  void SetUp() override {
    MagnetisationMonitorTest::SetUp();
    initialise_lattice_with_monitor(false);
    globals::solver = &solver_;
    auto spins = globals::s.mutable_host_view();
    for (int i = 0; i < globals::num_spins; ++i) {
      spins(i, 0) = 1.0;
      spins(i, 1) = spins(i, 2) = 0.0;
    }
  }

  void check_field_and_energy(Hamiltonian& h, double time, double field) {
    // The clock and phase remain double; the resulting field retains Real precision.
    const double expected = static_cast<jams::Real>(field) * globals::mus(0);
    const double tolerance = 8 * std::numeric_limits<jams::Real>::epsilon();
    EXPECT_NEAR(h.calculate_field(0, time)[0], expected, tolerance);
    EXPECT_NEAR(h.calculate_energy(0, time), -expected, tolerance);
  }

  MagnetisationStubSolver solver_;
};

TEST_F(TimeDependentFieldTest, SincCarrierResolvesLateStageTimes) {
  libconfig::Config config;
  config.readString(R"(ham = {
    module = "applied-field"; type = "sinc-cos"; field = [1.0, 0.0, 0.0];
    time_center = 1.6777216e-9; freq_bandwidth = 1.0e15; freq_center = 2.5e15;
  };)");
  AppliedFieldHamiltonian h(config.lookup("ham"), globals::num_spins);
  const double centre = 1.6777216e-9 / 1e-12;
  for (const double offset : {0.0, 0.00005, 0.0001, 0.0002}) {
    const double time = centre + offset;
    const double tau = time - centre;
    const double x = kPi * 1000.0 * tau;
    const double envelope = tau == 0 ? 1.0 : std::sin(x) / x;
    check_field_and_energy(h, time, envelope * std::cos(kTwoPi * 2500.0 * tau));
  }
}

TEST_F(TimeDependentFieldTest, GaussianCacheDistinguishesLateStageTimes) {
  libconfig::Config config;
  config.readString(R"(ham = {
    module = "field-pulse"; pulse_type = "gaussian_surface_pulse";
    surface_cutoff = -1.0; temporal_center = 1677.7216;
    temporal_width = 0.0001; field = [1.0, 0.0, 0.0];
  };)");
  FieldPulseHamiltonian h(config.lookup("ham"), globals::num_spins);
  for (const double offset : {0.0, 0.00005, 0.0001, 0.0002, 0.0001}) {
    const double time = 1677.7216 + offset;
    const double x = (time - 1677.7216) / 0.0001;
    check_field_and_energy(h, time, std::exp(-0.5 * x * x));
  }
}

TEST_F(TimeDependentFieldTest, SincPulseIsFiniteAtItsCentre) {
  libconfig::Config config;
  config.readString(R"(ham = {
    module = "field-pulse"; pulse_type = "sinc_pulse";
    temporal_center = 1677.7216; temporal_width = 10000.0;
    field = [1.0, 0.0, 0.0];
  };)");
  FieldPulseHamiltonian h(config.lookup("ham"), globals::num_spins);
  check_field_and_energy(h, 1677.7216, 1.0);
  const double time = 1677.7216 + 0.00005;
  const double x = 10000.0 * (time - 1677.7216);
  check_field_and_energy(h, time, std::sin(x) / x);
}

TEST_F(TimeDependentFieldTest, LocalAcFrequencyRetainsDoublePhasePrecision) {
  libconfig::Config config;
  config.readString(R"(ham = {
    module = "zeeman";
    ac_local_field = ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    ac_local_frequency = [1234.567890123, 1234.567890123];
  };)");
  ZeemanHamiltonian h(config.lookup("ham"), globals::num_spins);
  for (const double offset : {0.0, 0.00005, 0.0001}) {
    const double time = 1677.7216 + offset;
    check_field_and_energy(h, time, std::cos((kTwoPi * 1234.567890123) * time));
  }
}

#if HAS_CUDA
TEST_F(TimeDependentFieldTest, CudaSincCarrierResolvesLateStageTimes) {
  if (!magnetisation_cuda_device_available()) GTEST_SKIP() << "CUDA device unavailable";
  libconfig::Config config;
  config.readString(R"(ham = {
    module = "applied-field"; type = "sinc-cos"; field = [1.0, 0.0, 0.0];
    time_center = 1.6777216e-9; freq_bandwidth = 1.0e15; freq_center = 2.5e15;
  };)");
  CudaAppliedFieldHamiltonian h(config.lookup("ham"), globals::num_spins);
  const double centre = 1.6777216e-9 / 1e-12;
  for (double offset : {0.0, 0.00005, 0.0001}) {
    const double time = centre + offset;
    const double x = kPi * 1000.0 * (time - centre);
    const double envelope = x == 0 ? 1.0 : std::sin(x) / x;
    const double expected = static_cast<jams::Real>(envelope * std::cos(kTwoPi * 2500.0 * (time - centre))) * globals::mus(0);
    const auto& spins = solver_.spin_array_for_fields();
    h.calculate_fields(time, spins);
    h.calculate_energies(time, spins);
    ASSERT_EQ(cudaStreamSynchronize(h.get_stream()), cudaSuccess);
    EXPECT_NEAR(h.field(0, 0), expected, 8 * std::numeric_limits<jams::Real>::epsilon());
    EXPECT_NEAR(h.energy(0), -expected, 8 * std::numeric_limits<jams::Real>::epsilon());
  }
}

TEST_F(TimeDependentFieldTest, CudaGaussianResolvesLateStageTimes) {
  if (!magnetisation_cuda_device_available()) GTEST_SKIP() << "CUDA device unavailable";
  libconfig::Config config;
  config.readString(R"(ham = {
    module = "field-pulse"; surface_cutoff = -1.0;
    temporal_center = 1677.7216; temporal_width = 0.0001;
    field = [1.0, 0.0, 0.0];
  };)");
  CudaFieldPulseHamiltonian h(config.lookup("ham"), globals::num_spins);
  for (double offset : {0.0, 0.00005, 0.0001}) {
    const double time = 1677.7216 + offset;
    const double x = (time - 1677.7216) / 0.0001;
    const double expected = static_cast<jams::Real>(std::exp(-0.5 * x * x)) * globals::mus(0);
    h.calculate_fields(time, solver_.spin_array_for_fields());
    ASSERT_EQ(cudaStreamSynchronize(h.get_stream()), cudaSuccess);
    EXPECT_NEAR(h.field(0, 0), expected, 8 * std::numeric_limits<jams::Real>::epsilon());
  }
}
#endif

}  // namespace jams::testing

#endif
