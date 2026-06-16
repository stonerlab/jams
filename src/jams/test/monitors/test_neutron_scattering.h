#ifndef JAMS_TEST_MONITORS_TEST_NEUTRON_SCATTERING_H
#define JAMS_TEST_MONITORS_TEST_NEUTRON_SCATTERING_H

#include "gtest/gtest.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <libconfig.h++>

#include <jams/common.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/output.h>
#include <jams/monitors/neutron_scattering.h>

#if HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace jams::testing {

class NeutronScatteringStubSolver : public Solver {
public:
  NeutronScatteringStubSolver() { step_size_ = 0.25; }

  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "neutron-scattering-stub"; }
};

#if HAS_CUDA
class NeutronScatteringCudaStubSolver : public NeutronScatteringStubSolver {
public:
  bool is_cuda_solver() const override { return true; }
};

inline bool neutron_scattering_cuda_device_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

class NeutronScatteringMonitorTest : public ::testing::Test {
protected:
  using NeutronRows = std::vector<std::vector<double>>;

  void SetUp() override {
    globals::solver = nullptr;
    globals::lattice = nullptr;
    output_dir_ = std::filesystem::temp_directory_path()
        / ("jams_neutron_scattering_monitor_test_" + current_test_id());
    std::filesystem::remove_all(output_dir_);
  }

  void TearDown() override {
    globals::solver = nullptr;
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
    std::filesystem::remove_all(output_dir_);
  }

  NeutronRows run_neutron(
      Solver& solver,
      const std::string& run_name,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const std::string& estimator = "welch",
      const std::string& direct_sum_config = "",
      const bool include_outer_hkl_path = true) {
    initialise_lattice(
        spatial_backend,
        time_backend,
        estimator,
        direct_sum_config,
        include_outer_hkl_path);

    const auto run_dir = output_dir_ / run_name;
    std::filesystem::remove_all(run_dir);
    std::filesystem::create_directories(run_dir);
    jams::Jams::set_output_dir(run_dir.string());

    globals::solver = &solver;
    {
      NeutronScatteringMonitor monitor(first_monitor_settings());
      for (int t = 0; t < periodogram_length_; ++t) {
        write_spin_state(t);
        monitor.update(solver);
      }
    }
    globals::solver = nullptr;
    return read_neutron_rows();
  }

  NeutronRows run_two_basis_cancellation(
      Solver& solver,
      const std::string& run_name,
      const std::string& direct_sum_backend,
      const std::string& time_backend) {
    initialise_lattice_from_config_string(
        two_basis_cancellation_config(direct_sum_backend, time_backend));

    const auto run_dir = output_dir_ / run_name;
    std::filesystem::remove_all(run_dir);
    std::filesystem::create_directories(run_dir);
    jams::Jams::set_output_dir(run_dir.string());

    globals::solver = &solver;
    {
      NeutronScatteringMonitor monitor(first_monitor_settings());
      for (int t = 0; t < periodogram_length_; ++t) {
        write_two_basis_cancellation_spin_state(t);
        monitor.update(solver);
      }
    }
    globals::solver = nullptr;
    return read_neutron_rows();
  }

  void initialise_lattice(
      const std::string& spatial_backend,
      const std::string& time_backend,
      const std::string& estimator,
      const std::string& direct_sum_config = "",
      const bool include_outer_hkl_path = true) {
    initialise_lattice_from_config_string(config(
        spatial_backend,
        time_backend,
        estimator,
        direct_sum_config,
        include_outer_hkl_path));
  }

  void initialise_lattice_from_config_string(const std::string& config_string) {
    delete globals::lattice;
    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(config_string);
    globals::lattice->init_from_config(*globals::config);
  }

  const libconfig::Setting& first_monitor_settings() const {
    return globals::config->lookup("monitors")[0];
  }

  void write_spin_state(const int time_index) {
    auto spins = globals::s.mutable_host_view();
    for (int spin = 0; spin < globals::num_spins; ++spin) {
      const double site = static_cast<double>(spin);
      const double t = static_cast<double>(time_index);
      spins(spin, 0) = 0.10 * std::cos(0.37 * t + 0.13 * site);
      spins(spin, 1) = 0.07 * std::sin(0.51 * t + 0.19 * site);
      spins(spin, 2) = 1.0 + 0.03 * std::cos(0.23 * t + 0.29 * site);
    }
  }

  void write_two_basis_cancellation_spin_state(const int time_index) {
    auto spins = globals::s.mutable_host_view();
    const double t = static_cast<double>(time_index);
    const double sy = 0.20 * std::cos(0.37 * t) + 0.04 * std::sin(0.61 * t);
    const double sz = 0.12 * std::sin(0.43 * t);
    for (int spin = 0; spin < globals::num_spins; ++spin) {
      spins(spin, 0) = 0.0;
      spins(spin, 1) = sy;
      spins(spin, 2) = sz;
    }
  }

  static NeutronRows read_neutron_rows() {
    std::ifstream file(jams::output::monitor_filename_series("neutron-scattering_path", "tsv", 0));
    EXPECT_TRUE(file.good());

    NeutronRows rows;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream stream(line);
      std::vector<double> values;
      double value = 0.0;
      while (stream >> value) {
        values.push_back(value);
      }
      if (!values.empty()) {
        rows.push_back(std::move(values));
      }
    }
    return rows;
  }

  static void expect_rows_finite(const NeutronRows& rows) {
    ASSERT_EQ(rows.size(), 10u);
    for (const auto& row : rows) {
      ASSERT_EQ(row.size(), 12u);
      for (const double value : row) {
        EXPECT_TRUE(std::isfinite(value));
      }
    }
  }

  static void expect_rows_near(
      const NeutronRows& actual,
      const NeutronRows& expected,
      const double relative_tolerance = 1.0e-5,
      const double absolute_tolerance = 1.0e-8) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t row = 0; row < expected.size(); ++row) {
      ASSERT_EQ(actual[row].size(), expected[row].size()) << "row " << row;
      for (std::size_t col = 0; col < expected[row].size(); ++col) {
        const double scale = std::max(std::abs(expected[row][col]), 1.0);
        EXPECT_NEAR(actual[row][col], expected[row][col], absolute_tolerance + relative_tolerance * scale)
            << "row " << row << " col " << col;
      }
    }
  }

  static void expect_unpolarized_cross_section_zero(
      const NeutronRows& rows,
      const double tolerance = 1.0e-10) {
    ASSERT_EQ(rows.size(), 5u);
    for (const auto& row : rows) {
      ASSERT_EQ(row.size(), 12u);
      EXPECT_NEAR(row[10], 0.0, tolerance);
      EXPECT_NEAR(row[11], 0.0, tolerance);
    }
  }

  static std::string current_test_id() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string id = info
        ? std::string(info->test_suite_name()) + "_" + info->name()
        : "unknown";
    for (char& ch : id) {
      const auto uch = static_cast<unsigned char>(ch);
      if (!std::isalnum(uch) && ch != '_' && ch != '-') {
        ch = '_';
      }
    }
    return id;
  }

  static std::string periodogram_config(const std::string& estimator) {
    if (estimator == "welch") {
      return R"(
            estimator = "welch";
      )";
    }

    if (estimator == "multitaper") {
      return R"(
            estimator = "multitaper";
            multitaper_bandwidth = 2.0;
            multitaper_tapers = 2;
      )";
    }

    throw std::runtime_error("unexpected estimator in test");
  }

  static std::string config(
      const std::string& spatial_backend,
      const std::string& time_backend,
      const std::string& estimator,
      const std::string& direct_sum_config,
      const bool include_outer_hkl_path = true) {
    const std::string outer_hkl_path = include_outer_hkl_path ? R"(
          hkl_path = (
            [0.25, 0.0, 0.0],
            [0.50, 0.0, 0.0]
          );
      )" : "";
    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 2.5e-13;
        t_min  = 2.5e-13;
        t_max  = 2.0e-12;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.00, 0.0, 0.0])
        );
      };

      lattice : {
        size = [4, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };

      monitors = (
        {
          module = "neutron-scattering";
          output_steps = 1;
          keep_negative_frequencies = false;
          fftw_threads = 1;
          sk_time_series_backend = "memory";
          spatial_fft_backend = ")") + spatial_backend + R"(";
          time_fft_backend = ")" + time_backend + R"(";
      )" + outer_hkl_path + direct_sum_config + R"(
          compute_periodogram : {
            length = )" + std::to_string(periodogram_length_) + R"(;
            overlap = 0;
      )" + periodogram_config(estimator) + R"(
          };
        }
      );
    )";
  }

  static std::string two_basis_cancellation_config(
      const std::string& direct_sum_backend,
      const std::string& time_backend) {
    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 2.5e-13;
        t_min  = 2.5e-13;
        t_max  = 2.0e-12;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.0, 0.0, 0.0]),
          ("A", [0.5, 0.0, 0.0])
        );
      };

      lattice : {
        size = [1, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };

      monitors = (
        {
          module = "neutron-scattering";
          output_steps = 1;
          keep_negative_frequencies = false;
          fftw_threads = 1;
          sk_time_series_backend = "memory";
          spatial_fft_backend = "cpu";
          time_fft_backend = ")") + time_backend + R"(";
          direct_sum : {
            enabled = true;
            backend = ")" + direct_sum_backend + R"(";
            hkl_path = (
              [1.0, 0.0, 0.0]
            );
          };
          compute_periodogram : {
            length = )" + std::to_string(periodogram_length_) + R"(;
            overlap = 0;
            estimator = "welch";
          };
        }
      );
    )";
  }

  std::filesystem::path output_dir_;
  static constexpr int periodogram_length_ = 8;
};

TEST_F(NeutronScatteringMonitorTest, NonNegativeFrequencyOutputUsesRetainedBinsOnly) {
  NeutronScatteringStubSolver solver;
  const auto rows = run_neutron(solver, "cpu_welch", "cpu", "cpu");
  expect_rows_finite(rows);
}

TEST_F(NeutronScatteringMonitorTest, DirectSumExactQPathIsNotClampedToFftGrid) {
  NeutronScatteringStubSolver solver;
  const auto rows = run_neutron(
      solver,
      "direct_exact_q",
      "cpu",
      "cpu",
      "welch",
      R"(
          direct_sum : {
            enabled = true;
            backend = "cpu";
            hkl_path = (
              [0.30, 0.0, 0.0]
            );
          };
      )");

  ASSERT_EQ(rows.size(), 5u);
  for (const auto& row : rows) {
    ASSERT_EQ(row.size(), 12u);
    EXPECT_NEAR(row[2], 0.30, 1.0e-12);
    EXPECT_NEAR(row[3], 0.0, 1.0e-12);
    EXPECT_NEAR(row[4], 0.0, 1.0e-12);
  }
}

TEST_F(NeutronScatteringMonitorTest, DirectSumDoesNotRequireOuterHklPath) {
  NeutronScatteringStubSolver solver;
  const auto rows = run_neutron(
      solver,
      "direct_no_outer_hkl",
      "cpu",
      "cpu",
      "welch",
      R"(
          direct_sum : {
            enabled = true;
            backend = "cpu";
            hkl_path = (
              [0.30, 0.0, 0.0]
            );
          };
      )",
      false);

  ASSERT_EQ(rows.size(), 5u);
  for (const auto& row : rows) {
    ASSERT_EQ(row.size(), 12u);
    EXPECT_NEAR(row[2], 0.30, 1.0e-12);
  }
}

TEST_F(NeutronScatteringMonitorTest, DirectSumTwoBasisCancellationUsesFullPositionPhase) {
  NeutronScatteringStubSolver solver;
  const auto rows = run_two_basis_cancellation(
      solver,
      "direct_two_basis_phase_cancel",
      "cpu",
      "cpu");

  expect_unpolarized_cross_section_zero(rows);
}

TEST_F(NeutronScatteringMonitorTest, DirectSumCpuMatchesFftAtGridAlignedPoints) {
  NeutronScatteringStubSolver fft_solver;
  const auto fft_rows = run_neutron(fft_solver, "fft_grid_ref", "cpu", "cpu");

  NeutronScatteringStubSolver direct_solver;
  const auto direct_rows = run_neutron(
      direct_solver,
      "direct_grid",
      "cpu",
      "cpu",
      "welch",
      R"(
          direct_sum : {
            enabled = true;
            backend = "cpu";
            hkl_path = (
              [0.25, 0.0, 0.0],
              [0.50, 0.0, 0.0]
            );
            points_per_segment = 2;
          };
      )");

  expect_rows_near(direct_rows, fft_rows, 2.0e-5, 1.0e-7);
}

TEST_F(NeutronScatteringMonitorTest, DirectSumPathCountsIncludeEndpointsAndDedupeSharedNodes) {
  NeutronScatteringStubSolver solver;
  const auto rows = run_neutron(
      solver,
      "direct_endpoint_dedupe",
      "cpu",
      "cpu",
      "welch",
      R"(
          direct_sum : {
            enabled = true;
            backend = "cpu";
            hkl_path = (
              [0.00, 0.00, 0.0],
              [0.25, 0.00, 0.0],
              [0.25, 0.50, 0.0]
            );
            points_per_segment = [3, 3];
          };
      )");

  ASSERT_EQ(rows.size(), 25u);
  const std::vector<std::vector<double>> expected_hkl = {
      {0.000, 0.00, 0.0},
      {0.125, 0.00, 0.0},
      {0.250, 0.00, 0.0},
      {0.250, 0.25, 0.0},
      {0.250, 0.50, 0.0}};

  for (std::size_t i = 0; i < expected_hkl.size(); ++i) {
    EXPECT_NEAR(rows[i][2], expected_hkl[i][0], 1.0e-12);
    EXPECT_NEAR(rows[i][3], expected_hkl[i][1], 1.0e-12);
    EXPECT_NEAR(rows[i][4], expected_hkl[i][2], 1.0e-12);
  }
}

TEST_F(NeutronScatteringMonitorTest, DirectSumWindowDefaultsOriginToSimulationBoxCenter) {
  NeutronScatteringStubSolver solver;
  const auto rows = run_neutron(
      solver,
      "direct_y_window_default_center",
      "cpu",
      "cpu",
      "welch",
      R"(
          direct_sum : {
            enabled = true;
            backend = "cpu";
            hkl_path = (
              [0.25, 0.0, 0.0]
            );
            window : {
              y : { width = 0.5; };
            };
          };
      )");

  ASSERT_EQ(rows.size(), 5u);
  for (const auto& row : rows) {
    ASSERT_EQ(row.size(), 12u);
    EXPECT_NEAR(row[10], 0.0, 1.0e-12);
    EXPECT_NEAR(row[11], 0.0, 1.0e-12);
  }
}

TEST_F(NeutronScatteringMonitorTest, DirectSumWindowClippedBySimulationBoxThrows) {
  NeutronScatteringStubSolver solver;
  initialise_lattice(
      "cpu",
      "cpu",
      "welch",
      R"(
          direct_sum : {
            enabled = true;
            backend = "cpu";
            hkl_path = (
              [0.25, 0.0, 0.0]
            );
            window : {
              z : { origin = 100.0; width = 1.0; };
            };
          };
      )");

  std::string message;
  globals::solver = &solver;
  try {
    NeutronScatteringMonitor monitor(first_monitor_settings());
  } catch (const std::runtime_error& error) {
    message = error.what();
  }
  globals::solver = nullptr;

  ASSERT_FALSE(message.empty()) << "Expected clipped direct-sum spatial window to throw";
  EXPECT_NE(message.find("direct_sum.window.z"), std::string::npos);
  EXPECT_NE(message.find("clipped by the simulation box"), std::string::npos);
}

#if HAS_CUDA
TEST_F(NeutronScatteringMonitorTest, CudaSpatialCudaTimeMatchesCpuWelchRows) {
  if (!neutron_scattering_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  NeutronScatteringStubSolver cpu_solver;
  const auto cpu_rows = run_neutron(cpu_solver, "cpu_welch_ref", "cpu", "cpu");

  NeutronScatteringCudaStubSolver cuda_solver;
  const auto cuda_rows = run_neutron(cuda_solver, "cuda_time_welch", "cuda", "cuda");

  expect_rows_near(cuda_rows, cpu_rows);
}

TEST_F(NeutronScatteringMonitorTest, CudaSpatialAutoTimeMatchesCpuWelchRows) {
  if (!neutron_scattering_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  NeutronScatteringStubSolver cpu_solver;
  const auto cpu_rows = run_neutron(cpu_solver, "cpu_auto_ref", "cpu", "cpu");

  NeutronScatteringCudaStubSolver cuda_solver;
  const auto cuda_rows = run_neutron(cuda_solver, "cuda_time_auto", "cuda", "auto");

  expect_rows_near(cuda_rows, cpu_rows);
}

TEST_F(NeutronScatteringMonitorTest, CudaSpatialCudaTimeMatchesCpuMultitaperRows) {
  if (!neutron_scattering_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  NeutronScatteringStubSolver cpu_solver;
  const auto cpu_rows = run_neutron(cpu_solver, "cpu_multitaper_ref", "cpu", "cpu", "multitaper");

  NeutronScatteringCudaStubSolver cuda_solver;
  const auto cuda_rows = run_neutron(cuda_solver, "cuda_time_multitaper", "cuda", "cuda", "multitaper");

  expect_rows_near(cuda_rows, cpu_rows);
}

TEST_F(NeutronScatteringMonitorTest, DirectSumCudaSpatialCpuTimeMatchesCpuDirectRows) {
  if (!neutron_scattering_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  const std::string cpu_direct = R"(
          direct_sum : {
            enabled = true;
            backend = "cpu";
            hkl_path = (
              [0.25, 0.0, 0.0],
              [0.50, 0.0, 0.0]
            );
            points_per_segment = 2;
          };
      )";
  const std::string cuda_direct = R"(
          direct_sum : {
            enabled = true;
            backend = "cuda";
            hkl_path = (
              [0.25, 0.0, 0.0],
              [0.50, 0.0, 0.0]
            );
            points_per_segment = 2;
          };
      )";

  NeutronScatteringStubSolver cpu_solver;
  const auto cpu_rows = run_neutron(cpu_solver, "direct_cpu_ref", "cpu", "cpu", "welch", cpu_direct);

  NeutronScatteringCudaStubSolver cuda_solver;
  const auto cuda_rows = run_neutron(cuda_solver, "direct_cuda_cpu_time", "cpu", "cpu", "welch", cuda_direct);

  expect_rows_near(cuda_rows, cpu_rows, 2.0e-5, 1.0e-7);
}

TEST_F(NeutronScatteringMonitorTest, DirectSumCudaSpatialCudaTimeMatchesCpuDirectRows) {
  if (!neutron_scattering_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  const std::string cpu_direct = R"(
          direct_sum : {
            enabled = true;
            backend = "cpu";
            hkl_path = (
              [0.25, 0.0, 0.0],
              [0.50, 0.0, 0.0]
            );
            points_per_segment = 2;
          };
      )";
  const std::string cuda_direct = R"(
          direct_sum : {
            enabled = true;
            backend = "cuda";
            hkl_path = (
              [0.25, 0.0, 0.0],
              [0.50, 0.0, 0.0]
            );
            points_per_segment = 2;
          };
      )";

  NeutronScatteringStubSolver cpu_solver;
  const auto cpu_rows = run_neutron(cpu_solver, "direct_cpu_cuda_time_ref", "cpu", "cpu", "welch", cpu_direct);

  NeutronScatteringCudaStubSolver cuda_solver;
  const auto cuda_rows = run_neutron(cuda_solver, "direct_cuda_cuda_time", "cpu", "cuda", "welch", cuda_direct);

  expect_rows_near(cuda_rows, cpu_rows, 2.0e-5, 1.0e-7);
}

TEST_F(NeutronScatteringMonitorTest, DirectSumCudaSpatialCpuTimeTwoBasisCancellationUsesFullPositionPhase) {
  if (!neutron_scattering_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  NeutronScatteringCudaStubSolver solver;
  const auto rows = run_two_basis_cancellation(
      solver,
      "direct_cuda_cpu_time_two_basis_phase_cancel",
      "cuda",
      "cpu");

  expect_unpolarized_cross_section_zero(rows);
}

TEST_F(NeutronScatteringMonitorTest, DirectSumCudaSpatialCudaTimeTwoBasisCancellationUsesFullPositionPhase) {
  if (!neutron_scattering_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  NeutronScatteringCudaStubSolver solver;
  const auto rows = run_two_basis_cancellation(
      solver,
      "direct_cuda_cuda_time_two_basis_phase_cancel",
      "cuda",
      "cuda");

  expect_unpolarized_cross_section_zero(rows);
}
#endif

}  // namespace jams::testing

#endif  // JAMS_TEST_MONITORS_TEST_NEUTRON_SCATTERING_H
