#ifndef JAMS_TEST_MONITORS_TEST_MAGNETISATION_LAYERS_H
#define JAMS_TEST_MONITORS_TEST_MAGNETISATION_LAYERS_H

#include "gtest/gtest.h"

#include <filesystem>
#include <memory>
#include <string>

#include <libconfig.h++>

#include <jams/common.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/exception.h>
#include <jams/helpers/output.h>
#include <jams/interface/highfive.h>
#include <jams/monitors/magnetisation_layers.h>

#if HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace jams::testing {

class MagnetisationLayersStubSolver : public Solver {
public:
  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "magnetisation-layers-stub"; }
};

#if HAS_CUDA
class MagnetisationLayersCudaStubSolver : public MagnetisationLayersStubSolver {
public:
  bool is_cuda_solver() const override { return true; }
};

inline bool magnetisation_layers_cuda_device_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

class MagnetisationLayersMonitorTest : public ::testing::Test {
protected:
  void SetUp() override {
    globals::solver = nullptr;
    output_dir_ = std::filesystem::temp_directory_path() / "jams_magnetisation_layers_monitor_test";
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

  void initialise_lattice_with_monitor(const std::string& monitor_settings) {
    initialise_lattice_from_config(base_config() + monitor_settings);
  }

  void initialise_lattice_from_config(const std::string& config) {
    globals::config->readString(config);
    globals::lattice->init_from_config(*globals::config);
  }

  const libconfig::Setting& first_monitor_settings() const {
    return globals::config->lookup("monitors")[0];
  }

  static std::string base_config() {
    return R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 1.0e-16;
        t_min  = 1.0e-16;
        t_max  = 1.0e-16;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.0, 0.0, 0.0])
        );
      };

      lattice : {
        size = [1, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };
    )";
  }

  static std::vector<double> read_layer_positions() {
    HighFive::File file(
        jams::output::monitor_filename("magnetisation-layers", "h5"),
        HighFive::File::ReadOnly);
    std::vector<double> values;
    file.getDataSet("/jams/monitors/magnetisation-layers/groups/A/layer_positions").read(values);
    return values;
  }

  static std::vector<int> read_layer_spin_counts() {
    HighFive::File file(
        jams::output::monitor_filename("magnetisation-layers", "h5"),
        HighFive::File::ReadOnly);
    std::vector<int> values;
    file.getDataSet("/jams/monitors/magnetisation-layers/groups/A/layer_spin_count").read(values);
    return values;
  }

  static std::vector<double> read_first_update_magnetisation() {
    HighFive::File file(
        jams::output::monitor_filename("magnetisation-layers", "h5"),
        HighFive::File::ReadOnly);
    std::vector<std::vector<double>> rows;
    file.getDataSet("/jams/monitors/magnetisation-layers/timeseries/000000000/A/magnetisation").read(rows);

    std::vector<double> values;
    for (const auto& row : rows) {
      values.insert(values.end(), row.begin(), row.end());
    }
    return values;
  }

private:
  std::filesystem::path output_dir_;
};

TEST_F(MagnetisationLayersMonitorTest, RejectsZeroLayerNormal) {
  initialise_lattice_with_monitor(R"(
    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 0.0];
      }
    );
  )");

  EXPECT_THROW({
    MagnetisationLayersMonitor monitor(first_monitor_settings());
  }, jams::ConfigException);
}

TEST_F(MagnetisationLayersMonitorTest, RejectsNegativeLayerThickness) {
  initialise_lattice_with_monitor(R"(
    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
        layer_thickness = -1.0;
      }
    );
  )");

  EXPECT_THROW({
    MagnetisationLayersMonitor monitor(first_monitor_settings());
  }, jams::ConfigException);
}

TEST_F(MagnetisationLayersMonitorTest, RejectsNegativeDistanceTolerance) {
  initialise_lattice_with_monitor(R"(
    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
        distance_tolerance = -1.0e-4;
      }
    );
  )");

  EXPECT_THROW({
    MagnetisationLayersMonitor monitor(first_monitor_settings());
  }, jams::ConfigException);
}

TEST_F(MagnetisationLayersMonitorTest, DefaultDistanceToleranceScalesWithLatticeParameter) {
  initialise_lattice_from_config(R"(
    solver : {
      module = "llg-heun-cpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
    );

    unitcell : {
      symops = false;
      check_closeness = false;
      parameter = 2.0e-9;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0]),
        ("A", [0.0, 0.0, 0.000075])
      );
    };

    lattice : {
      size = [1, 1, 1];
      periodic = [true, true, true];
      normalise_spins = false;
    };

    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
      }
    );
  )");

  MagnetisationLayersMonitor monitor(first_monitor_settings());

  const auto layer_spin_counts = read_layer_spin_counts();
  ASSERT_EQ(layer_spin_counts.size(), 1u);
  EXPECT_EQ(layer_spin_counts[0], 2);
}

TEST_F(MagnetisationLayersMonitorTest, FiniteThicknessLayersUseStableBinCentres) {
  initialise_lattice_from_config(R"(
    solver : {
      module = "llg-heun-cpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
    );

    unitcell : {
      symops = false;
      parameter = 1.0e-9;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0])
      );
    };

    lattice : {
      size = [1, 1, 3];
      periodic = [true, true, true];
      normalise_spins = false;
    };

    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
        layer_thickness = 1.0;
      }
    );
  )");

  MagnetisationLayersMonitor monitor(first_monitor_settings());

  const auto layer_positions = read_layer_positions();
  ASSERT_EQ(layer_positions.size(), 3u);
  EXPECT_NEAR(layer_positions[0], 0.5, 1.0e-12);
  EXPECT_NEAR(layer_positions[1], 1.5, 1.0e-12);
  EXPECT_NEAR(layer_positions[2], 2.5, 1.0e-12);
}

TEST_F(MagnetisationLayersMonitorTest, FiniteThicknessLayersSnapBoundaryRoundoffWithinTolerance) {
  initialise_lattice_from_config(R"(
    solver : {
      module = "llg-heun-cpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
    );

    unitcell : {
      symops = false;
      check_closeness = false;
      parameter = 1.0e-9;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0]),
        ("A", [0.0, 0.0, 0.499999999999])
      );
    };

    lattice : {
      size = [1, 1, 1];
      periodic = [true, true, true];
      normalise_spins = false;
    };

    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
        layer_thickness = 0.5;
        distance_tolerance = 1.0e-9;
      }
    );
  )");

  MagnetisationLayersMonitor monitor(first_monitor_settings());

  const auto layer_positions = read_layer_positions();
  ASSERT_EQ(layer_positions.size(), 2u);
  EXPECT_NEAR(layer_positions[0], 0.25, 1.0e-12);
  EXPECT_NEAR(layer_positions[1], 0.75, 1.0e-12);

  const auto layer_spin_counts = read_layer_spin_counts();
  ASSERT_EQ(layer_spin_counts.size(), 2u);
  EXPECT_EQ(layer_spin_counts[0], 1);
  EXPECT_EQ(layer_spin_counts[1], 1);
}

TEST_F(MagnetisationLayersMonitorTest, ZeroThicknessLayersUseStrictMapWithToleranceLookup) {
  initialise_lattice_from_config(R"(
    solver : {
      module = "llg-heun-cpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
    );

    unitcell : {
      symops = false;
      check_closeness = false;
      parameter = 1.0e-9;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0]),
        ("A", [0.0, 0.0, 0.00005])
      );
    };

    lattice : {
      size = [1, 1, 1];
      periodic = [true, true, true];
      normalise_spins = false;
    };

    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
        distance_tolerance = 1.0e-4;
      }
    );
  )");

  MagnetisationLayersMonitor monitor(first_monitor_settings());

  const auto layer_spin_counts = read_layer_spin_counts();
  ASSERT_EQ(layer_spin_counts.size(), 1u);
  EXPECT_EQ(layer_spin_counts[0], 2);
}

TEST_F(MagnetisationLayersMonitorTest, UpdateAccumulatesLayerMagnetisationInOnePass) {
  initialise_lattice_from_config(R"(
    solver : {
      module = "llg-heun-cpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
    );

    unitcell : {
      symops = false;
      parameter = 1.0e-9;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0])
      );
    };

    lattice : {
      size = [1, 1, 2];
      periodic = [true, true, true];
      normalise_spins = false;
    };

    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
        layer_thickness = 1.0;
      }
    );
  )");

  MagnetisationLayersMonitor monitor(first_monitor_settings());
  MagnetisationLayersStubSolver solver;
  monitor.update(solver);

  const auto magnetisation = read_first_update_magnetisation();
  ASSERT_EQ(magnetisation.size(), 6u);
  EXPECT_NEAR(magnetisation[0], 1.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[1], 0.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[2], 0.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[3], 1.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[4], 0.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[5], 0.0, 1.0e-12);
}

#if HAS_CUDA
TEST_F(MagnetisationLayersMonitorTest, CudaFiniteThicknessUpdateMatchesExpectedLayerMagnetisation) {
  if (!magnetisation_layers_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  initialise_lattice_from_config(R"(
    solver : {
      module = "llg-heun-cuda";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
    );

    unitcell : {
      symops = false;
      parameter = 1.0e-9;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0])
      );
    };

    lattice : {
      size = [1, 1, 2];
      periodic = [true, true, true];
      normalise_spins = false;
    };

    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
        layer_thickness = 1.0;
      }
    );
  )");

  MagnetisationLayersCudaStubSolver solver;
  globals::solver = &solver;

  MagnetisationLayersMonitor monitor(first_monitor_settings());
  monitor.update(solver);

  const auto magnetisation = read_first_update_magnetisation();
  ASSERT_EQ(magnetisation.size(), 6u);
  EXPECT_NEAR(magnetisation[0], 1.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[1], 0.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[2], 0.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[3], 1.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[4], 0.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[5], 0.0, 1.0e-12);
}

TEST_F(MagnetisationLayersMonitorTest, CudaZeroThicknessOutputUsesExpectedLayerGrouping) {
  if (!magnetisation_layers_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  initialise_lattice_from_config(R"(
    solver : {
      module = "llg-heun-cuda";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
    );

    unitcell : {
      symops = false;
      check_closeness = false;
      parameter = 1.0e-9;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0]),
        ("A", [0.0, 0.0, 0.00005])
      );
    };

    lattice : {
      size = [1, 1, 1];
      periodic = [true, true, true];
      normalise_spins = false;
    };

    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
        distance_tolerance = 1.0e-4;
      }
    );
  )");

  MagnetisationLayersCudaStubSolver solver;
  globals::solver = &solver;

  MagnetisationLayersMonitor monitor(first_monitor_settings());
  monitor.update(solver);

  const auto layer_spin_counts = read_layer_spin_counts();
  ASSERT_EQ(layer_spin_counts.size(), 1u);
  EXPECT_EQ(layer_spin_counts[0], 2);

  const auto magnetisation = read_first_update_magnetisation();
  ASSERT_EQ(magnetisation.size(), 3u);
  EXPECT_NEAR(magnetisation[0], 2.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[1], 0.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[2], 0.0, 1.0e-12);
}

TEST_F(MagnetisationLayersMonitorTest, CudaChunkedReductionHandlesMoreThanOneChunkInLayer) {
  if (!magnetisation_layers_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  initialise_lattice_from_config(R"(
    solver : {
      module = "llg-heun-cuda";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [1.0, 0.0, 0.0]; }
    );

    unitcell : {
      symops = false;
      parameter = 1.0e-9;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0])
      );
    };

    lattice : {
      size = [257, 1, 1];
      periodic = [true, true, true];
      normalise_spins = false;
    };

    monitors = (
      {
        module = "magnetisation-layers";
        output_steps = 1;
        layer_normal = [0.0, 0.0, 1.0];
      }
    );
  )");

  MagnetisationLayersCudaStubSolver solver;
  globals::solver = &solver;

  MagnetisationLayersMonitor monitor(first_monitor_settings());
  monitor.update(solver);

  const auto layer_spin_counts = read_layer_spin_counts();
  ASSERT_EQ(layer_spin_counts.size(), 1u);
  EXPECT_EQ(layer_spin_counts[0], 257);

  const auto magnetisation = read_first_update_magnetisation();
  ASSERT_EQ(magnetisation.size(), 3u);
  EXPECT_NEAR(magnetisation[0], 257.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[1], 0.0, 1.0e-12);
  EXPECT_NEAR(magnetisation[2], 0.0, 1.0e-12);
}
#endif

}  // namespace jams::testing

#endif  // JAMS_TEST_MONITORS_TEST_MAGNETISATION_LAYERS_H
