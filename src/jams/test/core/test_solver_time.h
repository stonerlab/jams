#ifndef JAMS_TEST_CORE_TEST_SOLVER_TIME_H
#define JAMS_TEST_CORE_TEST_SOLVER_TIME_H

#include "gtest/gtest.h"
#include "jams/core/solver.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

namespace {

class TimeTestSolver : public Solver {
 public:
  void initialize(const libconfig::Setting& settings) override { initialize_time(settings); }
  void run() override { advance_time(); }
  std::string name() const override { return "time-test"; }
  int min_steps() const { return min_steps_; }
  void set_iteration(int iteration) {
    iteration_ = iteration;
    time_ = iteration_ * step_size_;
  }
  void set_stage(double offset) { time_ += offset; }
};

void initialise_test_time(TimeTestSolver& solver, double dt, double maximum, double minimum = 0.0) {
  libconfig::Config config;
  auto& settings = config.getRoot();
  settings.add("t_step", libconfig::Setting::TypeFloat) = dt;
  settings.add("t_max", libconfig::Setting::TypeFloat) = maximum;
  settings.add("t_min", libconfig::Setting::TypeFloat) = minimum;
  solver.initialize(settings);
}

TEST(SolverTime, KeepsAdjacentTimesAndStageOffsetsAtLargeIterations) {
  TimeTestSolver solver;
  static_assert(std::is_same_v<decltype(solver.time()), double>);
  initialise_test_time(solver, 1e-16, 1e-7);
  EXPECT_EQ(solver.time_step(), 1e-16 / 1e-12);
  for (const int n : {16777215, 16777216, 16777217, 1000000000}) {
    solver.set_iteration(n);
    const double start = solver.time();
    const double dt = solver.time_step();
    solver.set_stage(0.5 * dt);
    EXPECT_GT(solver.time(), start);
    EXPECT_LT(solver.time(), (n + 1) * dt);
    // Completing a step must discard its temporary stage time.
    solver.run();
    EXPECT_EQ(solver.time(), (n + 1) * dt);
    EXPECT_GT(solver.time(), start);
    EXPECT_EQ(solver.monitor_coordinates()[0], solver.time());
  }
}

TEST(SolverTime, SnapsIntegralRatiosAndRoundsDurationBounds) {
  TimeTestSolver solver;
  initialise_test_time(solver, 1e-16, 3e-16, 3e-16);
  EXPECT_EQ(solver.max_steps(), 3);
  EXPECT_EQ(solver.min_steps(), 3);
  initialise_test_time(solver, 1e-12, std::nextafter(3e-12, 0.0),
                       std::nextafter(2e-12, 3e-12));
  EXPECT_EQ(solver.max_steps(), 3);
  EXPECT_EQ(solver.min_steps(), 2);
  initialise_test_time(solver, 1e-12, 3.25e-12, 1.25e-12);
  EXPECT_EQ(solver.max_steps(), 3);
  EXPECT_EQ(solver.min_steps(), 2);
  initialise_test_time(solver, 1e-12, (3.0 - 1e-10) * 1e-12,
                       (1.0 + 1e-10) * 1e-12);
  EXPECT_EQ(solver.max_steps(), 2);
  EXPECT_EQ(solver.min_steps(), 2);
  initialise_test_time(solver, 1e-16, 0.0);
  EXPECT_EQ(solver.max_steps(), 0);
  EXPECT_FALSE(solver.is_running());
}

TEST(SolverTime, RejectsInvalidTimesAndCountsBeforeIntegerConversion) {
  TimeTestSolver solver;
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double dt : {0.0, -1e-16, inf, nan, std::numeric_limits<double>::max()}) {
    EXPECT_THROW(initialise_test_time(solver, dt, 1e-12), std::runtime_error);
  }
  for (double limit : {-1e-12, inf, nan}) {
    EXPECT_THROW(initialise_test_time(solver, 1e-16, limit), std::runtime_error);
    EXPECT_THROW(initialise_test_time(solver, 1e-16, 1e-12, limit), std::runtime_error);
  }
  EXPECT_THROW(initialise_test_time(solver, 1e-12, 1e-12, 2e-12), std::runtime_error);
  EXPECT_THROW(initialise_test_time(solver, 1e-12, 1.5e-12, 1.25e-12), std::runtime_error);
  EXPECT_THROW(initialise_test_time(solver, 1e-12, (double(std::numeric_limits<int>::max()) + 1) * 1e-12), std::runtime_error);
  EXPECT_THROW(initialise_test_time(solver, std::numeric_limits<double>::denorm_min(), 1.0), std::runtime_error);
  initialise_test_time(solver, 1e-12, double(std::numeric_limits<int>::max()) * 1e-12);
  EXPECT_EQ(solver.max_steps(), std::numeric_limits<int>::max());
}

TEST(SolverTime, RegularSamplingExcludesFinalCompletedStep) {
  TimeTestSolver solver;
  initialise_test_time(solver, 1e-16, 3e-16);
  std::vector<double> samples;
  while (solver.is_running()) {
    samples.push_back(solver.time());
    solver.run();
  }
  ASSERT_EQ(samples.size(), 3);
  EXPECT_EQ(samples[0], 0.0);
  EXPECT_EQ(samples[2], 2 * solver.time_step());
  EXPECT_EQ(solver.time(), 3 * solver.time_step());
}

TEST(TimeOutput, RoundTripsTimeIndependentlyOfObservablePrecision) {
  using namespace jams::output;
  const std::vector<ColDef> cols = {time_column(), {"energy", "meV", ColFmt::Scientific}};
  std::vector<std::string> printed_times;
  for (const double time : {1000000000 * 1e-4, 1000000001 * 1e-4, 1.2345678901234567e-9}) {
    std::ostringstream plain, optional;
    write_tsv_row(plain, cols, std::vector<double>{time, 1.23456789}, 3);
    write_tsv_row(optional, cols, std::vector<std::optional<double>>{time, 1.23456789}, 3);
    EXPECT_EQ(plain.str(), optional.str());
    std::istringstream row(plain.str());
    std::string time_text, energy_text;
    row >> time_text >> energy_text;
    EXPECT_EQ(std::stod(time_text), time);
    EXPECT_EQ(energy_text, "1.235e+00");
    printed_times.push_back(time_text);
    EXPECT_EQ(make_tsv_header_row(cols, 3).find('\t'), plain.str().find('\t'));
  }
  EXPECT_NE(printed_times[0], printed_times[1]);
  std::ostringstream missing;
  write_tsv_row(missing, cols, std::vector<std::optional<double>>{std::nullopt, 1.0}, 3);
  EXPECT_EQ(missing.str().find('\t'), make_tsv_header_row(cols, 3).find('\t'));
}

}  // namespace

#endif
