#ifndef JAMS_TEST_CORE_THERMOSTAT_TEMPERATURE_PROFILE_H
#define JAMS_TEST_CORE_THERMOSTAT_TEMPERATURE_PROFILE_H

#include "gtest/gtest.h"

#include <memory>
#include <string>

#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/thermostat_temperature_profile.h"
#include "jams/helpers/exception.h"

class ThermostatTemperatureProfileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    globals::config = std::make_unique<libconfig::Config>();
    globals::lattice = new Lattice();
  }

  void TearDown() override {
    delete globals::lattice;
    globals::lattice = nullptr;
    globals::config = nullptr;
  }

  void read_config(const std::string& thermostat_config) {
    read_config_from_base(base_config(), thermostat_config);
  }

  void read_config_from_base(
      const std::string& lattice_config,
      const std::string& thermostat_config) {
    globals::config->readString(lattice_config + thermostat_config);
    globals::lattice->init_from_config(*globals::config);
  }

  static std::string base_config() {
    return R"(
      materials = (
        {
          name = "A";
          moment = 1.0;
          alpha = 0.1;
          spin = [0.0, 0.0, 1.0];
        }
      );

      unitcell = {
        parameter = 1.0;
        basis = ([1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]));
      };

      lattice = {
        size = [4, 1, 1];
        periodic = [false, false, false];
      };
    )";
  }

  static std::string stretched_x_base_config() {
    return R"(
      materials = (
        {
          name = "A";
          moment = 1.0;
          alpha = 0.1;
          spin = [0.0, 0.0, 1.0];
        }
      );

      unitcell = {
        parameter = 1.0;
        basis = ([2.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]));
      };

      lattice = {
        size = [2, 1, 1];
        periodic = [false, false, false];
      };
    )";
  }
};

TEST_F(ThermostatTemperatureProfileTest, ParsesConstantAndLinearRegions) {
  read_config(R"(
    thermostat = {
      temperature_regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [2.0, 1.0, 1.0];
          temperature = 100.0;
        },
        {
          type = "linear";
          origin = [2.0, 0.0, 0.0];
          size = [2.0, 1.0, 1.0];
          direction = [1.0, 0.0, 0.0];
          low = 200.0;
          high = 400.0;
        }
      );
    };
  )");

  const auto profile = jams::ThermostatTemperatureProfile::from_config(
      *globals::config, globals::num_spins, 0.0);

  ASSERT_TRUE(profile.is_per_spin());
  ASSERT_EQ(profile.size(), 4);
  EXPECT_DOUBLE_EQ(profile.temperature()(0), 100.0);
  EXPECT_DOUBLE_EQ(profile.temperature()(1), 100.0);
  EXPECT_DOUBLE_EQ(profile.temperature()(2), 200.0);
  EXPECT_DOUBLE_EQ(profile.temperature()(3), 300.0);
  EXPECT_DOUBLE_EQ(profile.sqrt_temperature()(0), 10.0);
}

TEST_F(ThermostatTemperatureProfileTest, RejectsOverlappingRegions) {
  read_config(R"(
    thermostat = {
      temperature_regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [2.0, 1.0, 1.0];
          temperature = 100.0;
        },
        {
          type = "constant";
          origin = [1.0, 0.0, 0.0];
          size = [2.0, 1.0, 1.0];
          temperature = 200.0;
        }
      );
    };
  )");

  EXPECT_THROW(
      (void)jams::ThermostatTemperatureProfile::from_config(
          *globals::config, globals::num_spins, 0.0),
      jams::ConfigException);
}

TEST_F(ThermostatTemperatureProfileTest, RegionsDefaultToFractionalCoordinates) {
  read_config_from_base(stretched_x_base_config(), R"(
    thermostat = {
      temperature_regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          temperature = 100.0;
        },
        {
          type = "constant";
          origin = [1.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          temperature = 200.0;
        }
      );
    };
  )");

  const auto profile = jams::ThermostatTemperatureProfile::from_config(
      *globals::config, globals::num_spins, 0.0);

  ASSERT_TRUE(profile.is_per_spin());
  ASSERT_EQ(profile.size(), 2);
  EXPECT_DOUBLE_EQ(profile.temperature()(0), 100.0);
  EXPECT_DOUBLE_EQ(profile.temperature()(1), 200.0);
}

TEST_F(ThermostatTemperatureProfileTest, CartesianRegionsRemainSupported) {
  read_config_from_base(stretched_x_base_config(), R"(
    thermostat = {
      temperature_regions = (
        {
          type = "constant";
          coordinate_format = "cartesian";
          origin = [0.0, 0.0, 0.0];
          size = [2.0, 1.0, 1.0];
          temperature = 100.0;
        },
        {
          type = "constant";
          coordinate_format = "cartesian";
          origin = [2.0, 0.0, 0.0];
          size = [2.0, 1.0, 1.0];
          temperature = 200.0;
        }
      );
    };
  )");

  const auto profile = jams::ThermostatTemperatureProfile::from_config(
      *globals::config, globals::num_spins, 0.0);

  ASSERT_TRUE(profile.is_per_spin());
  ASSERT_EQ(profile.size(), 2);
  EXPECT_DOUBLE_EQ(profile.temperature()(0), 100.0);
  EXPECT_DOUBLE_EQ(profile.temperature()(1), 200.0);
}

#endif  // JAMS_TEST_CORE_THERMOSTAT_TEMPERATURE_PROFILE_H
