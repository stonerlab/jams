#ifndef JAMS_TEST_MONITORS_TEST_MAGNON_SPECTRUM_CUDA_H
#define JAMS_TEST_MONITORS_TEST_MAGNON_SPECTRUM_CUDA_H

#include "gtest/gtest.h"

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
#include <jams/monitors/magnon_spectrum.h>

#if HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace jams::testing {

class MagnonSpectrumStubSolver : public Solver {
public:
  MagnonSpectrumStubSolver() { step_size_ = 0.25; }

  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "magnon-spectrum-stub"; }
};

#if HAS_CUDA
class MagnonSpectrumCudaStubSolver : public MagnonSpectrumStubSolver {
public:
  bool is_cuda_solver() const override { return true; }
};

inline bool magnon_spectrum_cuda_device_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

class MagnonSpectrumCudaMonitorTest : public ::testing::Test {
protected:
  using SpectrumTable = std::vector<std::vector<double>>;

  void SetUp() override {
    globals::solver = nullptr;
    globals::lattice = nullptr;
    output_dir_ = std::filesystem::temp_directory_path()
        / ("jams_magnon_spectrum_monitor_test_" + current_test_id());
    std::filesystem::remove_all(output_dir_);
  }

  void TearDown() override {
    globals::solver = nullptr;
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
    std::filesystem::remove_all(output_dir_);
  }

  SpectrumTable run_spectrum(
      Solver& solver,
      const std::string& run_name,
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib = 0) {
    initialise_lattice(estimator, spatial_backend, time_backend, cuda_memory_limit_mib);

    const auto run_dir = output_dir_ / run_name;
    std::filesystem::remove_all(run_dir);
    std::filesystem::create_directories(run_dir);
    jams::Jams::set_output_dir(run_dir.string());

    globals::solver = &solver;
    {
      MagnonSpectrumMonitor monitor(first_monitor_settings());
      for (int t = 0; t < periodogram_length_; ++t) {
        write_spin_state(t);
        monitor.update(solver);
      }
    }
    globals::solver = nullptr;

    return read_spectrum_rows();
  }

  void initialise_lattice(
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib = 0) {
    delete globals::lattice;
    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(base_config(
        estimator,
        spatial_backend,
        time_backend,
        cuda_memory_limit_mib));
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
      const double phase = 0.77 * t + 0.31 * site;
      spins(spin, 0) = 0.07 * std::cos(phase) + 0.015 * std::sin(0.37 * t + 0.11 * site);
      spins(spin, 1) = 0.05 * std::sin(1.19 * phase + 0.23 * site);
      spins(spin, 2) = 1.0 + 0.02 * std::cos(0.41 * t + 0.17 * site);
    }
  }

  static SpectrumTable read_spectrum_rows() {
    std::ifstream file(jams::output::monitor_filename_series("magnon-spectrum_path", "tsv", 0, 1));
    EXPECT_TRUE(file.good());

    SpectrumTable rows;
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

  static void expect_spectra_near(
      const SpectrumTable& actual,
      const SpectrumTable& expected,
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

  static void expect_finite_spectrum(const SpectrumTable& rows) {
    ASSERT_FALSE(rows.empty());
    for (const auto& row : rows) {
      ASSERT_EQ(row.size(), 13u);
      for (const double value : row) {
        EXPECT_TRUE(std::isfinite(value));
      }
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

  static std::string base_config(
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib) {
    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 2.5e-13;
        t_min  = 2.5e-13;
        t_max  = 2.0e-12;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; },
        { name = "B"; moment = 1.5; spin = [0.0, 0.0, 1.0]; }
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
          ("B", [0.50, 0.0, 0.0])
        );
      };

      lattice : {
        size = [2, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };

      monitors = (
        {
          module = "magnon-spectrum";
          output_steps = 1;
          output_magnon_spectrum = true;
          site_resolved = false;
          keep_negative_frequencies = false;
          fftw_threads = 1;
          sk_time_series_backend = "memory";
          spatial_fft_backend = ")") + spatial_backend + R"(";
          time_fft_backend = ")" + time_backend + R"(";
          cuda_time_fft_memory_limit_mib = )" + std::to_string(cuda_memory_limit_mib) + R"(;
          hkl_path = (
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]
          );
          compute_periodogram : {
            length = )" + std::to_string(periodogram_length_) + R"(;
            overlap = 0;
      )" + periodogram_config(estimator) + R"(
          };
        }
      );
    )";
  }

  static constexpr int periodogram_length_ = 8;
  std::filesystem::path output_dir_;
};

TEST_F(MagnonSpectrumCudaMonitorTest, AutoBackendsWithCpuSolverProduceFiniteSpectrum) {
  MagnonSpectrumStubSolver solver;
  const auto rows = run_spectrum(solver, "cpu_auto", "welch", "auto", "auto");
  expect_finite_spectrum(rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, RejectsCudaTimeWithCpuSpatialBackend) {
  MagnonSpectrumStubSolver solver;
  initialise_lattice("welch", "cpu", "cuda");
  globals::solver = &solver;
  EXPECT_THROW((void)MagnonSpectrumMonitor(first_monitor_settings()), std::runtime_error);
  globals::solver = nullptr;
}

TEST_F(MagnonSpectrumCudaMonitorTest, RejectsExplicitCudaSpatialWithoutCudaSolver) {
  MagnonSpectrumStubSolver solver;
  initialise_lattice("welch", "cuda", "cpu");
  globals::solver = &solver;
  EXPECT_THROW((void)MagnonSpectrumMonitor(first_monitor_settings()), std::runtime_error);
  globals::solver = nullptr;
}

#if HAS_CUDA
TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCpuTimeMatchesCpuWelchSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_spectrum(cpu_solver, "cpu_welch", "welch", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_spectrum(cuda_solver, "cuda_spatial_cpu_time_welch", "welch", "cuda", "cpu");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuWelchSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_spectrum(cpu_solver, "cpu_welch_cuda_time_ref", "welch", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_spectrum(cuda_solver, "cuda_spatial_cuda_time_welch", "welch", "cuda", "cuda");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCpuTimeMatchesCpuMultitaperSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_spectrum(cpu_solver, "cpu_multitaper", "multitaper", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_spectrum(cuda_solver, "cuda_spatial_cpu_time_multitaper", "multitaper", "cuda", "cpu");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuMultitaperSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_spectrum(cpu_solver, "cpu_multitaper_cuda_time_ref", "multitaper", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_spectrum(cuda_solver, "cuda_spatial_cuda_time_multitaper", "multitaper", "cuda", "cuda");

  expect_spectra_near(cuda_rows, cpu_rows);
}
#endif

}  // namespace jams::testing

#endif  // JAMS_TEST_MONITORS_TEST_MAGNON_SPECTRUM_CUDA_H
