#ifndef JAMS_TEST_MONITORS_TEST_NEUTRON_SCATTERING_H
#define JAMS_TEST_MONITORS_TEST_NEUTRON_SCATTERING_H

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
#include <jams/monitors/neutron_scattering.h>

namespace jams::testing {

class NeutronScatteringStubSolver : public Solver {
public:
  NeutronScatteringStubSolver() { step_size_ = 0.25; }

  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "neutron-scattering-stub"; }
};

class NeutronScatteringMonitorTest : public ::testing::Test {
protected:
  void SetUp() override {
    globals::solver = nullptr;
    globals::lattice = nullptr;
    output_dir_ = std::filesystem::temp_directory_path() / "jams_neutron_scattering_monitor_test";
    std::filesystem::remove_all(output_dir_);
    std::filesystem::create_directories(output_dir_);
    jams::Jams::set_output_dir(output_dir_.string());
  }

  void TearDown() override {
    globals::solver = nullptr;
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
    std::filesystem::remove_all(output_dir_);
  }

  void initialise_lattice() {
    delete globals::lattice;
    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(config());
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

  static std::vector<std::vector<double>> read_neutron_rows() {
    std::ifstream file(jams::output::monitor_filename_series("neutron-scattering_path", "tsv", 0));
    EXPECT_TRUE(file.good());

    std::vector<std::vector<double>> rows;
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

  static std::string config() {
    return R"(
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
          spatial_fft_backend = "cpu";
          time_fft_backend = "cpu";
          hkl_path = (
            [0.25, 0.0, 0.0],
            [0.50, 0.0, 0.0]
          );
          compute_periodogram : {
            length = 8;
            overlap = 0;
            estimator = "welch";
          };
        }
      );
    )";
  }

  std::filesystem::path output_dir_;
};

TEST_F(NeutronScatteringMonitorTest, NonNegativeFrequencyOutputUsesRetainedBinsOnly) {
  initialise_lattice();

  NeutronScatteringStubSolver solver;
  globals::solver = &solver;

  {
    NeutronScatteringMonitor monitor(first_monitor_settings());
    for (int t = 0; t < 8; ++t) {
      write_spin_state(t);
      monitor.update(solver);
    }
  }
  globals::solver = nullptr;

  const auto rows = read_neutron_rows();
  ASSERT_EQ(rows.size(), 10u);
  for (const auto& row : rows) {
    ASSERT_EQ(row.size(), 12u);
    for (const double value : row) {
      EXPECT_TRUE(std::isfinite(value));
    }
  }
}

}  // namespace jams::testing

#endif  // JAMS_TEST_MONITORS_TEST_NEUTRON_SCATTERING_H
