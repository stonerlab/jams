#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>

#include <filesystem>
#include <string>
#include <type_traits>

#include "jams/common.h"
#include "jams/containers/vec3.h"
#include "jams/core/globals.h"
#include "jams/core/jams++.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_common.h"
#include "jams/hamiltonian/applied_field.h"

namespace {

using Spin = jams::Vec<double, 3>;

class CudaHeunTest : public testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
    old_output_ = jams::Jams::output_path();
    output_ = std::filesystem::temp_directory_path() /
              ("jams-cuda-heun-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  }

  void TearDown() override {
    if (!initialized_) return;
    jams::cleanup_simulation();
    globals::solver = nullptr;
    globals::lattice = nullptr;
    globals::config = nullptr;
    jams::Jams::set_mode(jams::Mode::CPU);
    jams::Jams::set_output_dir(old_output_);
    std::filesystem::remove_all(output_);
  }

  void initialize(int num_spins, double temperature, const char* sampling = "auto") {
    const std::string config = R"(
      materials = ({name = "A"; moment = 1.4; alpha = 0.2; spin = [1.0, 0.0, 0.0];});
      unitcell = {parameter = 3e-10;
        basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]));};
      lattice = {size = [)" + std::to_string(num_spins) +
                               R"(, 1, 1]; periodic = [false, false, false];};
      physics = {temperature = )" +
                               std::to_string(temperature) + R"(;};
      hamiltonians = ({module = "applied-field"; type = "sinc-cos";
        field = [0.0, 2.0, 8.0]; time_center = 0.0;
        freq_bandwidth = 1e12; freq_center = 2e12;});
      solver = {module = "llg-heun-gpu"; thermostat = "quantum-spde-gpu";
                thermostat_sampling = ")" + std::string(sampling) + R"(";
                t_step = 2e-14; t_max = 4e-14;};
      monitors = ();
    )";

    jams::ProgramArgs args;
    args.output_path = output_.string();
    args.config_inputs.push_back({config, true});
    jams::initialize_simulation(args);
    initialized_ = true;
  }

  void check_step(bool expect_thermal_field = false) {
    auto& solver = *globals::solver;
    const auto initial = globals::s;
    const double time = solver.time();
    const double dt = solver.time_step();
    const int iteration = solver.iteration();
    solver.update_physics_module();
    solver.run();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto& applied_field = static_cast<AppliedFieldHamiltonian&>(*solver.hamiltonians().front());
    const auto* noise = solver.thermostat()->data();
    if (expect_thermal_field) {
      EXPECT_TRUE(std::any_of(noise, noise + globals::num_spins3,
                              [](jams::Real value) { return value != 0.0; }));
    }
    const auto spins = globals::s.host_view();
    const auto ds_dt = globals::ds_dt.host_view();

    for (int i = 0; i < globals::num_spins; ++i) {
      SCOPED_TRACE(i);
      const Spin spin = {initial(i, 0), initial(i, 1), initial(i, 2)};
      const auto rhs = [&](const Spin& state, double stage_time) {
        const auto field = applied_field.calculate_field(i, stage_time);
        Spin effective_field;
        for (int component = 0; component < 3; ++component) {
          const jams::Real value =
              field[component] * globals::inv_mus(i) + noise[3 * i + component];
          effective_field[component] = value;
        }
        const auto precession = jams::cross(state, effective_field);
        const auto damping = jams::cross(state, precession);
        return -double(globals::gyro(i)) * (precession + double(globals::alpha(i)) * damping);
      };
      const auto k1 = rhs(spin, time);
      const auto stage_2_spin = jams::unit_vector(spin + dt * k1);
      // The same thermostat sample must be used in both stages.
      const auto k2 = rhs(stage_2_spin, time + dt);
      const auto expected_ds_dt = 0.5 * (k1 + k2);
      const auto expected = jams::unit_vector(spin + dt * expected_ds_dt);
      constexpr double tolerance = std::is_same_v<jams::Real, float> ? 2e-6 : 2e-12;
      for (int component = 0; component < 3; ++component) {
        EXPECT_NEAR(spins(i, component), expected[component], tolerance);
        EXPECT_NEAR(ds_dt(i, component), expected_ds_dt[component], tolerance);
      }
      const Spin actual = {spins(i, 0), spins(i, 1), spins(i, 2)};
      EXPECT_NEAR(jams::dot(actual, actual), jams::dot(spin, spin), 2e-12);
    }
    EXPECT_EQ(solver.iteration(), iteration + 1);
    EXPECT_EQ(solver.time(), (iteration + 1) * dt);
  }

 private:
  std::filesystem::path output_;
  std::string old_output_;
  bool initialized_ = false;
};

TEST_F(CudaHeunTest, SingleSpinUsesEndOfStepField) {
  initialize(1, 0.0);
  check_step();
  check_step();
}

TEST_F(CudaHeunTest, PartialBlockPreservesZeroSpinAndReusesThermalField) {
  initialize(257, 50.0, "legacy");
  // Set a true zero vector after lattice initialisation to exercise normalisation.
  globals::s(0, 0) = 0.0;
  check_step(true);
  check_step(true);
}

}  // namespace
