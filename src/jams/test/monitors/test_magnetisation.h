#ifndef JAMS_TEST_MONITORS_TEST_MAGNETISATION_H
#define JAMS_TEST_MONITORS_TEST_MAGNETISATION_H

#include "gtest/gtest.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <libconfig.h++>

#include <jams/common.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/output.h>
#include <jams/monitors/magnetisation.h>

#if HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace jams::testing {

class MagnetisationStubSolver : public Solver {
public:
  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "magnetisation-stub"; }
};

#if HAS_CUDA
class MagnetisationCudaStubSolver : public MagnetisationStubSolver {
public:
  bool is_cuda_solver() const override { return true; }
};

inline bool magnetisation_cuda_device_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

class MagnetisationMonitorTest : public ::testing::Test {
protected:
  void SetUp() override {
    globals::solver = nullptr;
    output_dir_ = std::filesystem::temp_directory_path() / "jams_magnetisation_monitor_test";
    std::filesystem::remove_all(output_dir_);
    jams::Jams::set_output_dir(output_dir_.string());

    globals::config = std::make_unique<libconfig::Config>();
    globals::lattice = new Lattice();
  }

  void TearDown() override {
    globals::solver = nullptr;
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
    std::filesystem::remove_all(output_dir_);
  }

  void initialise_lattice_with_monitor(const bool normalize) {
    globals::config->readString(base_config(normalize));
    globals::lattice->init_from_config(*globals::config);

    auto spins = globals::s.mutable_host_view();
    for (auto spin = 0; spin < globals::num_spins; ++spin) {
      spins(spin, 0) = 0.05 * static_cast<double>((spin % 11) - 5);
      spins(spin, 1) = 0.03 * static_cast<double>((spin % 7) - 3);
      spins(spin, 2) = 0.02 * static_cast<double>((spin % 5) - 2);
    }
  }

  const libconfig::Setting& first_monitor_settings() const {
    return globals::config->lookup("monitors")[0];
  }

  std::vector<double> run_monitor_update(Solver& solver) {
    globals::solver = &solver;
    {
      MagnetisationMonitor monitor(first_monitor_settings());
      monitor.update(solver);
    }
    return read_first_magnetisation_row();
  }

  static std::vector<double> read_first_magnetisation_row() {
    std::ifstream file(jams::output::monitor_filename("magnetisation", "tsv"));
    EXPECT_TRUE(file.good());

    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    std::vector<double> values;
    std::istringstream row(line);
    double value = 0.0;
    while (row >> value) {
      values.push_back(value);
    }
    return values;
  }

  static std::string base_config(const bool normalize) {
    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 1.0e-16;
        t_min  = 1.0e-16;
        t_max  = 1.0e-16;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; },
        { name = "B"; moment = 2.0; spin = [0.0, 1.0, 0.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.00, 0.0, 0.0]),
          ("B", [0.25, 0.0, 0.0]),
          ("A", [0.50, 0.0, 0.0])
        );
      };

      lattice : {
        size = [1, 1, 97];
        periodic = [true, true, true];
        normalise_spins = false;
      };

      monitors = (
        {
          module = "magnetisation";
          output_steps = 1;
          grouping = "none";
          normalize = )") + (normalize ? "true" : "false") + R"(;
          precision = 15;
        }
      );
    )";
  }

private:
  std::filesystem::path output_dir_;
};

TEST_F(MagnetisationMonitorTest, CpuUpdateWritesExpectedNormalisedTotalMagnetisation) {
  initialise_lattice_with_monitor(true);

  MagnetisationStubSolver solver;
  const auto values = run_monitor_update(solver);

  ASSERT_EQ(values.size(), 5u);
  EXPECT_NEAR(values[0], 0.0, 1.0e-15);
  EXPECT_TRUE(std::isfinite(values[1]));
  EXPECT_TRUE(std::isfinite(values[2]));
  EXPECT_TRUE(std::isfinite(values[3]));
  EXPECT_NEAR(values[4], std::sqrt(values[1] * values[1] + values[2] * values[2] + values[3] * values[3]), 1.0e-14);
}

#if HAS_CUDA
TEST_F(MagnetisationMonitorTest, CudaUpdateMatchesCpuNormalisedTotalMagnetisation) {
  if (!magnetisation_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  initialise_lattice_with_monitor(true);

  MagnetisationStubSolver cpu_solver;
  const auto cpu_values = run_monitor_update(cpu_solver);

  MagnetisationCudaStubSolver cuda_solver;
  const auto cuda_values = run_monitor_update(cuda_solver);

  ASSERT_EQ(cuda_values.size(), cpu_values.size());
  for (std::size_t i = 0; i < cpu_values.size(); ++i) {
    EXPECT_NEAR(cuda_values[i], cpu_values[i], 1.0e-8);
  }
}

TEST_F(MagnetisationMonitorTest, CudaUpdateMatchesCpuUnnormalisedTotalMagnetisation) {
  if (!magnetisation_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  initialise_lattice_with_monitor(false);

  MagnetisationStubSolver cpu_solver;
  const auto cpu_values = run_monitor_update(cpu_solver);

  MagnetisationCudaStubSolver cuda_solver;
  const auto cuda_values = run_monitor_update(cuda_solver);

  ASSERT_EQ(cuda_values.size(), cpu_values.size());
  for (std::size_t i = 0; i < cpu_values.size(); ++i) {
    EXPECT_NEAR(cuda_values[i], cpu_values[i], 1.0e-6);
  }
}
#endif

}  // namespace jams::testing

#endif  // JAMS_TEST_MONITORS_TEST_MAGNETISATION_H
