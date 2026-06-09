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
#include <jams/helpers/exception.h>
#include <jams/monitors/magnetisation_layers.h>

namespace jams::testing {

class MagnetisationLayersMonitorTest : public ::testing::Test {
protected:
  void SetUp() override {
    output_dir_ = std::filesystem::temp_directory_path() / "jams_magnetisation_layers_monitor_test";
    std::filesystem::remove_all(output_dir_);
    jams::Jams::set_output_dir(output_dir_.string());

    globals::config = std::make_unique<libconfig::Config>();
    globals::lattice = new Lattice();
  }

  void TearDown() override {
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
    std::filesystem::remove_all(output_dir_);
  }

  void initialise_lattice_with_monitor(const std::string& monitor_settings) {
    globals::config->readString(base_config() + monitor_settings);
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

}  // namespace jams::testing

#endif  // JAMS_TEST_MONITORS_TEST_MAGNETISATION_LAYERS_H
