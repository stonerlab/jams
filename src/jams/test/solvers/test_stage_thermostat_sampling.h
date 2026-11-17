#ifndef JAMS_TEST_STAGE_THERMOSTAT_SAMPLING_H
#define JAMS_TEST_STAGE_THERMOSTAT_SAMPLING_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/jams++.h"
#include "jams/core/solver.h"
#include "jams/solvers/llg_rkmk_functions.h"

namespace {

constexpr double kStageTestField = 6.0;
constexpr double kStageTestFrequency = 1.25;

class RotatingStageBath final : public Thermostat {
 public:
  RotatingStageBath(double tick_interval, int ticks, bool gpu)
      : Thermostat(0.0, 0.0, tick_interval, 1), gpu_(gpu) {
#if HAS_CUDA
    if (gpu_) cuda_stream_ = CudaStream(CudaStream::Priority::LOW);
#endif
    for (int tick = 0; tick <= ticks; ++tick) {
      const double angle = kStageTestFrequency * tick_interval * tick;
      samples_.push_back({static_cast<jams::Real>(kStageTestField * std::cos(angle)),
                          static_cast<jams::Real>(kStageTestField * std::sin(angle)), 0.0});
    }
  }

  void sample_initial() override {
    ++initial_calls;
    update();
  }

  void update() override {
    const auto& field = samples_.at(published++);
#if HAS_CUDA
    if (gpu_) {
      CHECK_CUDA_STATUS(cudaMemcpyAsync(noise_.mutable_device_data(), field.data(),
          sizeof(field), cudaMemcpyHostToDevice, get_stream()));
      return;
    }
#endif
    for (int n = 0; n < 3; ++n) noise_(0, n) = field[n];
  }

  int initial_calls = 0;
  int published = 0;

 private:
  [[maybe_unused]] bool gpu_;
  std::vector<std::array<jams::Real, 3>> samples_;
};

class StageCountingHamiltonian final : public Hamiltonian {
 public:
  explicit StageCountingHamiltonian(bool gpu) : gpu_(gpu) {
    set_name("stage-test-counting-field");
    field_.resize(1, 3).zero();
#if HAS_CUDA
    if (gpu_) {
      cuda_stream_ = CudaStream(CudaStream::Priority::LOW);
      CHECK_CUDA_STATUS(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    }
#endif
  }

  ~StageCountingHamiltonian() override {
#if HAS_CUDA
    if (gpu_) cudaEventDestroy(done_);
#endif
  }

  void calculate_fields(double time, const SpinArray&) override {
    times.push_back(time);
  }

  jams::Vec<jams::Real, 3> calculate_field(int, double) override { return {}; }
  jams::Real calculate_energy(int, double) override { return 0.0; }

  std::vector<double> times;

 private:
  [[maybe_unused]] bool gpu_;
};

struct StageSamplingCase {
  std::string_view solver;
  int legacy_ticks;
  int stage_ticks;
  std::vector<double> stages;

  bool gpu() const { return solver.ends_with("-gpu"); }
  bool rk4() const { return stages.size() == 4; }
  bool additive() const { return solver == "gse-rk4-gpu"; }
};

const StageSamplingCase stage_sampling_cases[] = {
    {"llg-heun-cpu", 1, 1, {0.0, 1.0}},
    {"llg-rkmk2-cpu", 2, 2, {0.0, 0.5}},
    {"llg-rkmk4-cpu", 2, 2, {0.0, 0.5, 0.5, 1.0}},
#if HAS_CUDA
    {"llg-heun-gpu", 1, 1, {0.0, 1.0}},
    {"llg-rkmk2-gpu", 2, 2, {0.0, 0.5}},
    {"llg-rkmk4-gpu", 2, 2, {0.0, 0.5, 0.5, 1.0}},
    {"llg-dm-gpu", 2, 2, {0.0, 1.0}},
    {"llg-simp-gpu", 2, 2, {0.0, 0.5}},
    {"llg-rk4-gpu", 1, 2, {0.0, 0.5, 0.5, 1.0}},
    {"llg-sot-rk4-gpu", 1, 2, {0.0, 0.5, 0.5, 1.0}},
    {"gse-rk4-gpu", 1, 2, {0.0, 0.5, 0.5, 1.0}},
#endif
};

class StageThermostatSampling : public ::testing::TestWithParam<StageSamplingCase> {
 protected:
  void SetUp() override {
#if HAS_CUDA
    int devices = 0;
    if (GetParam().gpu() && (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)) GTEST_SKIP();
#endif
    old_output_ = jams::Jams::output_path();
    output_ = std::filesystem::temp_directory_path() /
        ("jams-thermostat-stage-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  }

  void TearDown() override {
    if (initialized_) finish_run();
    if (!output_.empty()) {
      jams::Jams::set_output_dir(old_output_);
      std::filesystem::remove_all(output_);
    }
  }

  void initialize(int steps, const std::string& sampling = {}) {
    const std::string config = R"(
      sim = {seed = 1234567890;};
      materials = ({name = "A"; moment = 2.0; alpha = 0.0; spin = [0.0, 0.0, 1.0];});
      unitcell = {parameter = 3e-10;
        basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]));};
      lattice = {size = [1, 1, 1]; periodic = [false, false, false];};
      physics = {temperature = 50.0;};
      hamiltonians = ();
      monitors = ();
    )";
    const std::string backend = GetParam().gpu() ? "gpu" : "cpu";
    const std::string sampling_setting = sampling.empty()
        ? "" : "thermostat_sampling = \"" + sampling + "\";";
    const std::string sot = GetParam().solver == "llg-sot-rk4-gpu"
        ? "spin_polarisation = [0.0, 1.0, 0.0]; spin_hall_angle = 0.0; charge_current_density = 0.0;"
        : "";
    const std::string solver = "solver = {module = \"" + std::string(GetParam().solver)
        + "\"; thermostat = \"quantum-spde-" + backend + "\";" + sampling_setting + "t_step = "
        + std::to_string(1.0 / steps) + "e-12; t_max = 1e-12;" + sot + "};";
    jams::ProgramArgs args;
    args.output_path = output_.string();
    args.config_inputs.push_back({config + solver, true});
    jams::initialize_simulation(args);
    initialized_ = true;
  }

  double rotating_field_error(int steps, bool count_fields) {
    initialize(steps);
    auto& solver = *globals::solver;
    const double h = solver.time_step();
    const auto& test = GetParam();
    auto* bath = new RotatingStageBath(h / test.stage_ticks, test.stage_ticks * steps, test.gpu());
    solver.register_thermostat(bath);
    StageCountingHamiltonian* hamiltonian = nullptr;
    if (count_fields) {
      hamiltonian = new StageCountingHamiltonian(test.gpu());
      solver.register_hamiltonian(hamiltonian);
    }

    while (solver.is_running()) {
      solver.update_physics_module();
      solver.run();
    }
    synchronize();
    EXPECT_EQ(bath->published, test.stage_ticks * steps + 1);
    EXPECT_EQ(bath->initial_calls, 1);
    EXPECT_EQ(solver.iteration(), steps);
    if (hamiltonian) {
      const auto count = test.stages.size();
      EXPECT_EQ(hamiltonian->times.size(), count * steps);
      for (std::size_t stage = 0; stage < hamiltonian->times.size(); ++stage) {
        const double expected_time = (stage / count + test.stages[stage % count]) * h;
        EXPECT_NEAR(hamiltonian->times[stage], expected_time, 1e-14);
      }
    }

    const double time = solver.time();
    const double gamma_field = globals::gyro(0) * kStageTestField;
    const double angle = kStageTestFrequency * time;
    double expected[3];
    if (test.additive()) {
      expected[0] = gamma_field * std::sin(angle) / kStageTestFrequency;
      expected[1] = gamma_field * (1.0 - std::cos(angle)) / kStageTestFrequency;
      expected[2] = 1.0;
    } else {
      const double start[3] = {0.0, 0.0, 1.0};
      const double frame_rate[3] = {gamma_field * time, 0.0, -angle};
      const double lab_rate[3] = {0.0, 0.0, angle};
      double frame_spin[3];
      jams::solvers::rkmk::rodrigues_rotate(frame_rate, start, frame_spin);
      jams::solvers::rkmk::rodrigues_rotate(lab_rate, frame_spin, expected);
    }
    double error_squared = 0.0;
    double norm_squared = 0.0;
    for (int n = 0; n < 3; ++n) {
      const double spin = globals::s(0, n);
      error_squared += (spin - expected[n]) * (spin - expected[n]);
      norm_squared += spin * spin;
    }
    if (!test.additive()) EXPECT_NEAR(norm_squared, 1.0, 2e-12);
    finish_run();
    return std::sqrt(error_squared);
  }

  void synchronize() {
#if HAS_CUDA
    if (GetParam().gpu()) CHECK_CUDA_STATUS(cudaDeviceSynchronize());
#endif
  }

 private:
  void finish_run() {
    synchronize();
    jams::cleanup_simulation();
    globals::solver = nullptr;
    globals::lattice = nullptr;
    globals::config = nullptr;
    jams::Jams::set_mode(jams::Mode::CPU);
    initialized_ = false;
  }

  bool initialized_ = false;
  std::filesystem::path output_;
  std::filesystem::path old_output_;
};

INSTANTIATE_TEST_SUITE_P(Solvers, StageThermostatSampling, ::testing::ValuesIn(stage_sampling_cases),
    [](const ::testing::TestParamInfo<StageSamplingCase>& info) {
      std::string name(info.param.solver);
      std::replace(name.begin(), name.end(), '-', '_');
      return name;
    });

TEST_P(StageThermostatSampling, UsesStageTimesAndIntegratesARotatingField) {
  const double coarse = rotating_field_error(4, true);
  const double fine = rotating_field_error(8, false);
  EXPECT_LT(coarse, GetParam().rk4() ? 1e-3 : 5e-2);
  EXPECT_LT(fine, coarse / (GetParam().rk4() ? 8.0 : 3.0));
}

TEST_P(StageThermostatSampling, LegacyKeepsItsUpdateCadence) {
  initialize(4, "legacy");
  auto& solver = *globals::solver;
  const auto& test = GetParam();
  auto* bath = new RotatingStageBath(solver.time_step() / test.legacy_ticks,
                                   4 * test.legacy_ticks, test.gpu());
  solver.register_thermostat(bath);
  while (solver.is_running()) {
    solver.update_physics_module();
    solver.run();
  }
  synchronize();
  EXPECT_EQ(bath->published, 4 * test.legacy_ticks);
  EXPECT_EQ(bath->initial_calls, 0);
}

}  // namespace

#endif  // JAMS_TEST_STAGE_THERMOSTAT_SAMPLING_H
