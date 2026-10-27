#ifndef JAMS_TEST_INITIALIZER_TEST_SPIN_PERTURBATION_H
#define JAMS_TEST_INITIALIZER_TEST_SPIN_PERTURBATION_H

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <vector>

#include <jams/common.h>
#include <jams/core/globals.h>
#include <jams/helpers/finite.h>
#include <jams/initializer/init_dispatcher.h>
#include <jams/interface/highfive.h>

namespace jams::testing {

class SpinPerturbationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    globals::num_spins = 64;
    globals::num_spins3 = 3 * globals::num_spins;
    globals::s.resize(globals::num_spins, 3);
    globals::mus.resize(globals::num_spins);
    reset_spins();
  }
  void TearDown() override {
    globals::s.clear();
    globals::mus.clear();
    globals::num_spins = globals::num_spins3 = 0;
  }
  void reset_spins() {
    for (int i = 0; i < globals::num_spins; ++i) {
      globals::mus(i) = 1.0;
      // Different local directions and lengths, including the negative axes.
      for (int j = 0; j < 3; ++j) globals::s(i, j) = 0.0;
      globals::s(i, i % 3) = (i % 2 ? -1.0 : 1.0) * (1.0 + i / 64.0);
    }
  }
  void execute(double amplitude = 1e-3, int seed = 1234) {
    libconfig::Config config;
    config.readString("initializer = {module = \"spin-perturbation\"; amplitude = 0.0; seed = 0;};");
    config.lookup("initializer.amplitude") = amplitude;
    config.lookup("initializer.seed") = seed;
    InitializerDispatcher::execute(config.lookup("initializer"));
  }
  std::vector<double> snapshot() const {
    const auto values = globals::s.host_span();
    return {values.begin(), values.end()};
  }
};

TEST_F(SpinPerturbationTest, PreservesLengthsAndBoundsLocalAngles) {
  // Include a non-axis-aligned, non-unit spin.
  globals::s(0, 0) = 0.6;
  globals::s(0, 1) = 0.8;
  globals::s(0, 2) = 1.5;
  const auto before = snapshot();
  const auto rng_before = Jams::random_generator_internal_state();
  execute();
  EXPECT_EQ(Jams::random_generator_internal_state(), rng_before);
  const auto after = snapshot();
  for (int i = 0; i < globals::num_spins; ++i) {
    const Vec<double, 3> a = {before[3*i], before[3*i+1], before[3*i+2]};
    const Vec<double, 3> b = {after[3*i], after[3*i+1], after[3*i+2]};
    EXPECT_NEAR(norm(a), norm(b), 1e-14);
    const auto n = a / norm(a);
    const double parallel = dot(n, b);
    const double transverse = norm(b - parallel * n);
    EXPECT_GT(parallel, 0.0);
    EXPECT_GT(transverse, 0.0);
    EXPECT_LE(transverse / parallel, 1e-3 + 1e-14);
  }
  // Identical starting spins receive independent draws.
  for (int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 1.0;
  }
  execute();
  EXPECT_NE(globals::s(0, 0), globals::s(1, 0));
}

TEST_F(SpinPerturbationTest, ReproducibleLocalSeedAndExactZeroAmplitude) {
  const auto original = snapshot();
  const auto rng_before = Jams::random_generator_internal_state();
  execute(0.0);
  EXPECT_EQ(snapshot(), original);
  execute();
  const auto first = snapshot();
  reset_spins();
  execute();
  EXPECT_EQ(snapshot(), first);
  reset_spins();
  execute(1e-3, 5678);
  EXPECT_NE(snapshot(), first);
  EXPECT_EQ(Jams::random_generator_internal_state(), rng_before);
}

TEST_F(SpinPerturbationTest, VacanciesAreUntouchedAndConsumeNoDraws) {
  for (int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 1.0;
  }
  const auto original = snapshot();
  execute();
  const auto all_magnetic = snapshot();
  std::copy(original.begin(), original.end(), globals::s.mutable_host_span().begin());
  globals::mus(0) = 0.0;
  execute();
  const auto first = snapshot();
  for (int j = 0; j < 3; ++j) {
    EXPECT_EQ(first[j], original[j]);
    EXPECT_EQ(first[3+j], all_magnetic[j]);
  }
  std::copy(original.begin(), original.end(), globals::s.mutable_host_span().begin());
  globals::s(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_NO_THROW(execute());
  EXPECT_FALSE(is_finite(globals::s(0, 0)));
  for (int i = 3; i < globals::num_spins3; ++i) EXPECT_EQ(globals::s.data()[i], first[i]);
}

TEST_F(SpinPerturbationTest, RejectsInvalidSettingsAndMagneticSpins) {
  for (double bad : {-1.0, std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(execute(bad), ConfigException);
  }
  EXPECT_THROW(execute(1e-3, -1), ConfigException);
  for (const auto* seed : {"1.5", "2147483648L"}) {
    libconfig::Config config;
    config.readString(std::string("initializer = {module = \"spin-perturbation\"; amplitude = 0.001; seed = ") + seed + ";};");
    EXPECT_THROW(InitializerDispatcher::execute(config.lookup("initializer")), ConfigException);
  }
  for (double bad : {0.0, std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()}) {
    reset_spins();
    globals::s(0, 0) = bad;
    EXPECT_THROW(execute(), ConfigException);
  }
}

TEST_F(SpinPerturbationTest, ComposesAfterH5InInitializerList) {
  const auto path = std::filesystem::temp_directory_path() /
      ("jams-perturb-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".h5");
  {
    HighFive::File file(path.string(), HighFive::File::ReadWrite | HighFive::File::Create);
    file.createDataSet<double>("/spins", HighFive::DataSpace({64, 3})).write(globals::s);
  }
  execute();
  const auto expected = snapshot();
  reset_spins();
  globals::s(0, 0) = 3.0;
  libconfig::Config config;
  config.readString("initializer = ({module = \"h5\"; spins = \"" + path.string() +
      "\";}, {module = \"spin-perturbation\"; amplitude = 0.001; seed = 1234;});");
  InitializerDispatcher::execute(config.lookup("initializer"));
  EXPECT_EQ(snapshot(), expected);
  std::filesystem::remove(path);
}

}  // namespace jams::testing
#endif
