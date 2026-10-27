#ifndef JAMS_TEST_MONITORS_SPIN_RATE_H
#define JAMS_TEST_MONITORS_SPIN_RATE_H

#include "gtest/gtest.h"
#include <filesystem>
#include <chrono>
#include <limits>
#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/monitors/spin_rate.h"
#include "jams/solvers/llg_spin_rate.h"
#if HAS_CUDA
#include "jams/cuda/cuda_solver.h"
#include "jams/core/jams++.h"
#endif

namespace jams::testing {

TEST(ScalarThreshold, CountsFreshSamplesAndResetsWithoutLatching) {
  ScalarThreshold threshold(1.0, 2);
  EXPECT_FALSE(threshold.satisfied());
  threshold.update(0.5);
  EXPECT_FALSE(threshold.satisfied());
  EXPECT_FALSE(threshold.satisfied());
  threshold.update(0.5);
  EXPECT_TRUE(threshold.satisfied());
  threshold.update(1.0);
  EXPECT_FALSE(threshold.satisfied());
  threshold.update(0.5);
  threshold.reset();
  threshold.update(0.5);
  EXPECT_FALSE(threshold.satisfied());
  threshold.update(0.5);
  EXPECT_TRUE(threshold.satisfied());
  EXPECT_THROW(threshold.update(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);
  EXPECT_FALSE(threshold.satisfied());
}

TEST(ScalarThreshold, RejectsInvalidConfiguration) {
  for (double threshold : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(ScalarThreshold(threshold, 1), std::runtime_error);
  }
  EXPECT_THROW(ScalarThreshold(1.0, 0), std::runtime_error);
  EXPECT_THROW(ScalarThreshold(1.0, -1), std::runtime_error);
}

TEST(SpinRateMath, AnalyticalRateFieldScalingAndGilbertPrefactor) {
  const double s[3] = {1.0, 0.0, 0.0};
  const double aligned[3] = {0.0, 0.0, 1.0};
  const double field[3] = {0.0, 0.0, 8.0};
  for (double alpha : {0.0, 0.25, 2.0}) {
    for (bool gilbert : {false, true}) {
      const double gyro = 3.0 / (gilbert ? 1.0 + alpha * alpha : 1.0);
      const double expected = gyro * 2.0 * std::sqrt(1.0 + alpha * alpha);
      EXPECT_NEAR(solvers::llg_spin_rate(s, field, 0.25, alpha, gyro), expected, 1e-13);
      EXPECT_EQ(solvers::llg_spin_rate(aligned, field, 0.25, alpha, gyro), 0.0);
    }
  }
  const double opposite[3] = {-1.0, 0.0, 0.0};
  const double a = solvers::llg_spin_rate(s, field, 0.25, 0.0, 3.0);
  const double b = solvers::llg_spin_rate(opposite, field, 0.25, 0.0, 3.0);
  EXPECT_EQ(solvers::combine_spin_rates(a, b), 6.0);
}

TEST(SpinRateMath, VacanciesZeroFieldAndInvalidValues) {
  double s[3] = {1.0, 0.0, 0.0};
  double field[3] = {0.0, 0.0, 1.0};
  EXPECT_EQ(solvers::llg_spin_rate<double>(s, nullptr, 1.0, 0.1, 1.0), 0.0);
  for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::infinity()}) {
    EXPECT_EQ(solvers::llg_spin_rate(s, field, bad, 0.1, 1.0), -1.0);
    EXPECT_EQ(solvers::llg_spin_rate(s, field, 1.0, bad, 1.0), -1.0);
    EXPECT_EQ(solvers::llg_spin_rate(s, field, 1.0, 0.1, bad), -1.0);
    field[0] = bad;
    EXPECT_EQ(solvers::llg_spin_rate(s, field, 1.0, 0.1, 1.0), -1.0);
    field[0] = 0.0;
    s[0] = bad;
    EXPECT_EQ(solvers::llg_spin_rate(s, field, 1.0, 0.1, 1.0), -1.0);
    EXPECT_EQ(solvers::llg_spin_rate(s, field, 0.0, bad, bad), 0.0);
    s[0] = 1.0;
  }
  EXPECT_EQ(solvers::combine_spin_rates(-1.0, 100.0), -1.0);
  EXPECT_EQ(solvers::combine_spin_rates(100.0, -1.0), -1.0);
  EXPECT_EQ(solvers::llg_spin_rate(s, field, 1.0, 0.1,
                                  std::numeric_limits<double>::max()), -1.0);
}

class SpinRateStubSolver : public Solver {
 public:
  void initialize(const libconfig::Setting&) override {}
  void run() override { advance_time(); }
  std::string name() const override { return "spin-rate-test"; }
  bool supports_deterministic_spin_rate() const override { return supported; }
  double max_deterministic_spin_rate() override { ++measurements; return rate; }
  void set_state(int iteration, int minimum = 0) {
    iteration_ = iteration;
    min_steps_ = minimum;
    time_ = iteration * step_size_;
  }
  bool supported = true;
  double rate = 0.5;
  int measurements = 0;
};

class SpinRateMonitorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    old_output_dir_ = jams::output::full_path_filename("");
    directory_ = std::filesystem::temp_directory_path() /
        ("jams-spin-rate-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory_);
    jams::Jams::set_output_dir(directory_.string());
    globals::solver = &solver_;
    old_magnetic_count_ = globals::num_magnetic_spins;
    globals::num_magnetic_spins = 1;
  }
  void TearDown() override {
    globals::solver = nullptr;
    globals::num_magnetic_spins = old_magnetic_count_;
    jams::Jams::set_output_dir(old_output_dir_);
    std::filesystem::remove_all(directory_);
  }
  std::unique_ptr<SpinRateMonitor> monitor(const std::string& extra) {
    libconfig::Config config;
    config.readString(("monitor = {module = \"spin-rate\"; " + extra + "};").c_str());
    return std::make_unique<SpinRateMonitor>(config.lookup("monitor"));
  }
  SpinRateStubSolver solver_;
  std::filesystem::path directory_;
  std::string old_output_dir_;
  int old_magnetic_count_;
};

TEST_F(SpinRateMonitorTest, MinimumTimeCadenceAndStatusQueries) {
  auto m = monitor("output_steps = 2; convergence = 1.0; consecutive_samples = 2;");
  auto* observed = m.get();
  solver_.register_monitor(m.release());
  for (int iteration = 0; iteration < 7; ++iteration) {
    solver_.set_state(iteration, 3);
    solver_.notify_monitors();
    const auto expected = iteration == 6 ? Monitor::ConvergenceStatus::kConverged
                                        : Monitor::ConvergenceStatus::kNotConverged;
    EXPECT_EQ(observed->convergence_status(), expected);
    EXPECT_EQ(solver_.convergence_status(), expected);
    EXPECT_EQ(solver_.convergence_status(), expected);
  }
  EXPECT_EQ(solver_.measurements, 4);
  solver_.rate = 2.0;
  solver_.set_state(8, 3);
  solver_.notify_monitors();
  EXPECT_EQ(solver_.convergence_status(), Monitor::ConvergenceStatus::kNotConverged);
}

TEST_F(SpinRateMonitorTest, DisabledAndInitialStateAndDelay) {
  auto disabled = monitor("output_steps = 1;");
  disabled->update(solver_);
  EXPECT_EQ(disabled->convergence_status(), Monitor::ConvergenceStatus::kDisabled);
  disabled.reset();
  auto enabled = monitor("output_steps = 2; start_delay_steps = 3; convergence = 1.0;");
  EXPECT_FALSE(enabled->is_updating(0));
  EXPECT_FALSE(enabled->is_updating(3));
  EXPECT_TRUE(enabled->is_updating(4));
  EXPECT_EQ(enabled->convergence_status(), Monitor::ConvergenceStatus::kNotConverged);
  enabled.reset();
  auto initial = monitor("convergence = 1.0;");
  initial->update(solver_);
  EXPECT_EQ(initial->convergence_status(), Monitor::ConvergenceStatus::kConverged);
}

TEST_F(SpinRateMonitorTest, PreservesAndSemantics) {
  auto first = monitor("name = \"first\"; output_steps = 1; convergence = 1.0;");
  auto second = monitor("name = \"second\"; output_steps = 1; convergence = 0.1;");
  solver_.register_monitor(first.release());
  solver_.register_monitor(second.release());
  solver_.notify_monitors();
  EXPECT_EQ(solver_.convergence_status(), Monitor::ConvergenceStatus::kNotConverged);
  solver_.rate = 0.01;
  solver_.notify_monitors();
  EXPECT_EQ(solver_.convergence_status(), Monitor::ConvergenceStatus::kConverged);
}

TEST_F(SpinRateMonitorTest, RejectsUnsupportedEmptyAndInvalidSettings) {
  for (const auto* extra : {"output_steps = 0;", "start_delay_steps = -1;",
                            "consecutive_samples = 0;", "convergence = 0.0;"}) {
    EXPECT_THROW(monitor(extra), std::runtime_error);
  }
  solver_.supported = false;
  EXPECT_THROW(monitor(""), std::runtime_error);
  solver_.supported = true;
  globals::num_magnetic_spins = 0;
  EXPECT_THROW(monitor(""), std::runtime_error);
}

class SpinRateCpuDiagnostic : public Solver {
 public:
  static constexpr bool gpu = false;
  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "spin-rate-cpu-test"; }
  double evaluate(const jams::MultiArray<jams::Real, 1>& gyro) { return max_llg_spin_rate_cpu(gyro); }
};

#if HAS_CUDA
class SpinRateCudaDiagnostic : public CudaSolver {
 public:
  static constexpr bool gpu = true;
  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "spin-rate-cuda-test"; }
  double evaluate(const jams::MultiArray<jams::Real, 1>& gyro) { return max_llg_spin_rate_cuda(gyro); }
};
#endif

template <typename Backend>
class SpinRateReductionTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if HAS_CUDA
    if constexpr (Backend::gpu) {
      int devices = 0;
      if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
      jams::Jams::set_mode(jams::Mode::GPU);
    }
#endif
    solver_ = std::make_unique<Backend>();
    globals::solver = solver_.get();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(R"(
      solver = {module = "llg-heun-cpu"; t_step = 1e-15; t_max = 1e-14;};
      materials = ({name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0];});
      unitcell = {parameter = 3e-10;
        basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]));};
      lattice = {size = [513, 1, 1]; periodic = [false, false, false];};
    )");
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
    // More than two CUDA blocks, including a partially filled last block.
    globals::num_spins = 513;
    globals::num_spins3 = 3 * globals::num_spins;
    globals::s.resize(513, 3);
    globals::h.resize(513, 3);
    globals::mus.resize(513);
    globals::alpha.resize(513);
    gyro_.resize(513);
    for (int i = 0; i < 513; ++i) {
      globals::s(i, 0) = (i % 2 == 0) ? 1.0 : -1.0;
      globals::s(i, 1) = globals::s(i, 2) = 0.0;
      globals::mus(i) = (i % 2 == 0) ? 2.0 : 4.0;
      globals::alpha(i) = 0.0;
      gyro_(i) = 1.0;
    }
    globals::mus(256) = 0.0;
    globals::sync_magnetic_moment_data();
    globals::alpha(512) = 0.75;
    gyro_(512) = 3.0;
    libconfig::Config config;
    config.readString("ham = {module = \"applied-field\"; field = [0.0, 0.0, 2.0];};");
    solver_->register_hamiltonian(Hamiltonian::create(config.lookup("ham"), 513, Backend::gpu));
  }
  void TearDown() override {
    solver_.reset();
    globals::solver = nullptr;
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
    globals::s.clear();
    globals::h.clear();
    globals::mus.clear();
    globals::inv_mus.clear();
    globals::alpha.clear();
    globals::gyro.clear();
    globals::ds_dt.clear();
    globals::positions.clear();
    gyro_.clear();
    globals::num_spins = globals::num_spins3 = globals::num_magnetic_spins = 0;
#if HAS_CUDA
    if constexpr (Backend::gpu) jams::Jams::set_mode(jams::Mode::CPU);
#endif
  }
  std::unique_ptr<Backend> solver_;
  jams::MultiArray<jams::Real, 1> gyro_;
};

#if HAS_CUDA
using SpinRateBackends = ::testing::Types<SpinRateCpuDiagnostic, SpinRateCudaDiagnostic>;
#else
using SpinRateBackends = ::testing::Types<SpinRateCpuDiagnostic>;
#endif
TYPED_TEST_SUITE(SpinRateReductionTest, SpinRateBackends);

TYPED_TEST(SpinRateReductionTest, MaximumAcrossBlocksAndVacanciesAndRepeatedCalls) {
  EXPECT_NEAR(this->solver_->evaluate(this->gyro_), 7.5, 1e-6);
  // A vacancy must not contribute even if its integration parameters are invalid.
  this->gyro_(256) = std::numeric_limits<jams::Real>::infinity();
  EXPECT_NEAR(this->solver_->evaluate(this->gyro_), 7.5, 1e-6);
  // Re-read current spins, not a stale field-spin cache or previous maximum.
  globals::s(512, 0) = 0.0;
  globals::s(512, 2) = 1.0;
  if constexpr (TypeParam::gpu) {
#if HAS_CUDA
    this->solver_->record_spin_barrier_event();
#endif
  }
  EXPECT_NEAR(this->solver_->evaluate(this->gyro_), 2.0, 1e-6);
}

TYPED_TEST(SpinRateReductionTest, InvalidValuesPropagateAndEmptySelectionFails) {
  for (int i : {0, 257, 512}) {
    const auto saved = this->gyro_(i);
    this->gyro_(i) = std::numeric_limits<jams::Real>::quiet_NaN();
    EXPECT_THROW(this->solver_->evaluate(this->gyro_), std::runtime_error);
    this->gyro_(i) = saved;
  }
  globals::num_magnetic_spins = 0;
  EXPECT_THROW(this->solver_->evaluate(this->gyro_), std::runtime_error);
}

TYPED_TEST(SpinRateReductionTest, NoHamiltoniansIgnorePreviouslyComputedField) {
  EXPECT_GT(this->solver_->evaluate(this->gyro_), 0.0);
  this->solver_->hamiltonians().clear();
  EXPECT_EQ(this->solver_->evaluate(this->gyro_), 0.0);
}

#if HAS_CUDA
TEST(SpinRateCudaTrajectories, PreservesExplicitlySeededThermalDynamics) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  const auto old_output = jams::Jams::output_path();
  const auto directory = std::filesystem::temp_directory_path() /
      ("jams-spin-rate-trajectory-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto trajectory = [&](const std::string& solver_name, bool diagnostic) {
    const std::string config = R"(
      sim = {seed = 1234567890;};
      materials = ({name = "A"; moment = 1.0; alpha = 0.1; spin = [1.0, 0.0, 0.0];});
      unitcell = {parameter = 3e-10;
        basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]));};
      lattice = {size = [2, 1, 1]; periodic = [false, false, false];};
      physics = {temperature = 50.0;};
      hamiltonians = (
        {module = "exchange"; energy_units = "meV";
         interactions = (("A", "A", [1.0, 0.0, 0.0], 1.0));},
        {module = "applied-field"; type = "sinc-cos"; field = [0.0, 0.0, 100.0];
         time_center = 0.0; freq_bandwidth = 1e12; freq_center = 2e12;}
      );
    )" "solver = {module = \"" + solver_name + "\"; t_step = 1e-15; t_max = 1e-14;};" +
        (diagnostic ? "monitors = ({module = \"spin-rate\"; output_steps = 1;});" : "monitors = ();");
    jams::ProgramArgs args;
    args.output_path = directory.string();
    args.config_inputs.push_back({config, true});
    jams::initialize_simulation(args);
    // The application currently creates cuRAND before applying sim.seed.
    // Explicitly seed here to isolate diagnostic effects from that behaviour.
    auto generator = jams::instance().curand_generator();
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());
    CHECK_CURAND_STATUS(curandSetStream(generator, nullptr));
    CHECK_CURAND_STATUS(curandSetPseudoRandomGeneratorSeed(generator, 1234567890));
    CHECK_CURAND_STATUS(curandSetGeneratorOffset(generator, 0));
    CHECK_CURAND_STATUS(curandGenerateSeeds(generator));

    std::vector<double> result;
    while (globals::solver->is_running()) {
      globals::solver->update_physics_module();
      globals::solver->notify_monitors();
      globals::solver->run();
      CHECK_CUDA_STATUS(cudaDeviceSynchronize());
      const auto spins = globals::s.host_span();
      for (int i = 0; i < globals::num_spins; ++i) {
        const auto* s = spins.data() + 3 * i;
        EXPECT_TRUE(jams::is_finite(s[0]) && jams::is_finite(s[1]) && jams::is_finite(s[2]));
        EXPECT_NEAR(s[0]*s[0] + s[1]*s[1] + s[2]*s[2], 1.0, 1e-12);
      }
      result.insert(result.end(), spins.begin(), spins.end());
    }
    const auto random_state = jams::Jams::random_generator_internal_state();
    jams::cleanup_simulation();
    globals::solver = nullptr;
    globals::lattice = nullptr;
    globals::config = nullptr;
    jams::Jams::set_mode(jams::Mode::CPU);
    return std::make_pair(result, random_state);
  };
  for (const auto* solver : {"llg-heun-gpu", "llg-rkmk2-gpu", "llg-rkmk4-gpu", "llg-rk4-gpu"}) {
    SCOPED_TRACE(solver);
    const auto baseline = trajectory(solver, false);
    EXPECT_EQ(baseline, trajectory(solver, false));
    EXPECT_EQ(baseline, trajectory(solver, true));
  }
  jams::Jams::set_output_dir(old_output);
  std::filesystem::remove_all(directory);
}
#endif

}  // namespace jams::testing
#endif
