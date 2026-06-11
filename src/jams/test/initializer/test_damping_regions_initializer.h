#ifndef JAMS_TEST_INITIALIZER_TEST_DAMPING_REGIONS_INITIALIZER_H
#define JAMS_TEST_INITIALIZER_TEST_DAMPING_REGIONS_INITIALIZER_H

#include "gtest/gtest.h"

#include <memory>
#include <string>

#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/utils.h"
#include "jams/initializer/init_dispatcher.h"

namespace {

constexpr double kAlphaTolerance = 1.0e-8;

void reset_damping_region_initializer_globals() {
  globals::num_spins = 0;
  globals::num_spins3 = 0;
  jams::util::force_deallocation(globals::s);
  jams::util::force_deallocation(globals::h);
  jams::util::force_deallocation(globals::ds_dt);
  jams::util::force_deallocation(globals::positions);
  jams::util::force_deallocation(globals::alpha);
  jams::util::force_deallocation(globals::mus);
  jams::util::force_deallocation(globals::gyro);
  globals::config = nullptr;
  globals::solver = nullptr;
  if (globals::lattice != nullptr) {
    delete globals::lattice;
    globals::lattice = nullptr;
  }
}

}  // namespace

class DampingRegionsInitializerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reset_damping_region_initializer_globals();
  }

  void TearDown() override {
    reset_damping_region_initializer_globals();
  }

  void read_config(const std::string& initializer_config) {
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(base_config() + initializer_config);
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
  }

  void execute_initializer() {
    jams::InitializerDispatcher::execute(globals::config->lookup("initializer"));
  }

  static std::string base_config() {
    return R"(
      materials = (
        {
          name = "A";
          moment = 1.0;
          alpha = 0.1;
          spin = [0.0, 0.0, 1.0];
        },
        {
          name = "B";
          moment = 1.0;
          alpha = 0.2;
          spin = [0.0, 0.0, 1.0];
        }
      );

      unitcell = {
        symops = false;
        parameter = 1.0;
        basis = ([1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]),
                     ("B", [0.5, 0.0, 0.0]));
      };

      lattice = {
        size = [4, 1, 1];
        periodic = [false, false, false];
      };
    )";
  }
};

TEST_F(DampingRegionsInitializerTest, ConstantRegionAssignsAlphaAndLeavesUnmatchedSpins) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          alpha = 0.05;
        }
      );
    };
  )");

  execute_initializer();

  EXPECT_NEAR(globals::alpha(0), 0.05, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(1), 0.05, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(2), 0.1, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(3), 0.2, kAlphaTolerance);
}

TEST_F(DampingRegionsInitializerTest, LinearRegionInterpolatesAlphaAlongDirection) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "linear";
          origin = [0.0, 0.0, 0.0];
          size = [2.0, 1.0, 1.0];
          direction = [1.0, 0.0, 0.0];
          low = 0.02;
          high = 0.10;
        }
      );
    };
  )");

  execute_initializer();

  EXPECT_NEAR(globals::alpha(0), 0.02, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(1), 0.04, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(2), 0.06, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(3), 0.08, kAlphaTolerance);
}

TEST_F(DampingRegionsInitializerTest, MaterialSelectorsAssignDistinctAlphaValues) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          materials = (
            { name = "A"; alpha = 0.03; },
            { name = "B"; alpha = 0.08; }
          );
        }
      );
    };
  )");

  execute_initializer();

  EXPECT_NEAR(globals::alpha(0), 0.03, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(1), 0.08, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(2), 0.1, kAlphaTolerance);
}

TEST_F(DampingRegionsInitializerTest, UnitCellPositionSelectorsUseOneBasedInputIndexing) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "linear";
          origin = [0.0, 0.0, 0.0];
          size = [2.0, 1.0, 1.0];
          direction = [1.0, 0.0, 0.0];
          unit_cell_positions = (
            { index = 1; low = 0.01; high = 0.05; },
            { index = 2; low = 0.02; high = 0.10; }
          );
        }
      );
    };
  )");

  execute_initializer();

  EXPECT_NEAR(globals::alpha(0), 0.01, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(1), 0.04, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(2), 0.03, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(3), 0.08, kAlphaTolerance);
}

TEST_F(DampingRegionsInitializerTest, RejectsDuplicateSpinAssignmentAcrossRegions) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          alpha = 0.03;
        },
        {
          type = "constant";
          origin = [0.5, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          alpha = 0.04;
        }
      );
    };
  )");

  EXPECT_THROW(execute_initializer(), jams::ConfigException);
}

TEST_F(DampingRegionsInitializerTest, RejectsInvalidMaterialName) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          materials = (
            { name = "C"; alpha = 0.03; }
          );
        }
      );
    };
  )");

  EXPECT_THROW(execute_initializer(), jams::ConfigException);
}

TEST_F(DampingRegionsInitializerTest, RejectsInvalidOneBasedUnitCellPositionIndex) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          unit_cell_positions = (
            { index = 0; alpha = 0.03; }
          );
        }
      );
    };
  )");

  EXPECT_THROW(execute_initializer(), jams::ConfigException);
}

TEST_F(DampingRegionsInitializerTest, RejectsMissingAlphaFields) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
        }
      );
    };
  )");

  EXPECT_THROW(execute_initializer(), jams::ConfigException);
}

TEST_F(DampingRegionsInitializerTest, RejectsAmbiguousAlphaModes) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          alpha = 0.03;
          materials = (
            { name = "A"; alpha = 0.04; }
          );
        }
      );
    };
  )");

  EXPECT_THROW(execute_initializer(), jams::ConfigException);
}

TEST_F(DampingRegionsInitializerTest, RejectsDuplicateUnitCellPositionSelectors) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          unit_cell_positions = (
            { index = 1; alpha = 0.03; },
            { index = 1; alpha = 0.04; }
          );
        }
      );
    };
  )");

  EXPECT_THROW(execute_initializer(), jams::ConfigException);
}

TEST_F(DampingRegionsInitializerTest, RejectsNegativeAlpha) {
  read_config(R"(
    initializer : {
      module = "damping-regions";
      regions = (
        {
          type = "constant";
          origin = [0.0, 0.0, 0.0];
          size = [1.0, 1.0, 1.0];
          alpha = -0.01;
        }
      );
    };
  )");

  EXPECT_THROW(execute_initializer(), jams::ConfigException);
}

TEST_F(DampingRegionsInitializerTest, InitializerListExecutesInOrder) {
  read_config(R"(
    initializer = (
      {
        module = "damping-regions";
        regions = (
          {
            type = "constant";
            origin = [0.0, 0.0, 0.0];
            size = [1.0, 1.0, 1.0];
            alpha = 0.03;
          }
        );
      },
      {
        module = "damping-regions";
        regions = (
          {
            type = "constant";
            origin = [0.0, 0.0, 0.0];
            size = [1.0, 1.0, 1.0];
            alpha = 0.07;
          }
        );
      }
    );
  )");

  execute_initializer();

  EXPECT_NEAR(globals::alpha(0), 0.07, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(1), 0.07, kAlphaTolerance);
}

TEST_F(DampingRegionsInitializerTest, LegacyUnnamedInitializerPathIsAccepted) {
  read_config(R"(
    initializer : {
    };
  )");

  EXPECT_NO_THROW(execute_initializer());
  EXPECT_NEAR(globals::alpha(0), 0.1, kAlphaTolerance);
  EXPECT_NEAR(globals::alpha(1), 0.2, kAlphaTolerance);
}

#endif  // JAMS_TEST_INITIALIZER_TEST_DAMPING_REGIONS_INITIALIZER_H
