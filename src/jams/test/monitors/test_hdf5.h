#ifndef JAMS_TEST_MONITORS_TEST_HDF5_H
#define JAMS_TEST_MONITORS_TEST_HDF5_H

#include "gtest/gtest.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <libconfig.h++>

#include <jams/common.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/physics.h>
#include <jams/core/solver.h>
#include <jams/core/thermostat.h>
#include <jams/helpers/output.h>
#include <jams/helpers/utils.h>
#include <jams/initializer/init_dispatcher.h>
#include <jams/interface/highfive.h>
#include <jams/monitors/hdf5.h>

namespace {

void reset_hdf5_monitor_globals() {
  globals::num_spins = 0;
  globals::num_spins3 = 0;
  jams::util::force_deallocation(globals::s);
  jams::util::force_deallocation(globals::h);
  jams::util::force_deallocation(globals::ds_dt);
  jams::util::force_deallocation(globals::positions);
  jams::util::force_deallocation(globals::alpha);
  jams::util::force_deallocation(globals::mus);
  jams::util::force_deallocation(globals::inv_mus);
  globals::num_magnetic_spins = 0;
  jams::util::force_deallocation(globals::gyro);
  globals::config = nullptr;
  globals::solver = nullptr;
  if (globals::lattice != nullptr) {
    delete globals::lattice;
    globals::lattice = nullptr;
  }
}

}  // namespace

namespace jams::testing {

class Hdf5MonitorStubSolver : public Solver {
 public:
  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "hdf5-monitor-stub"; }
};

class Hdf5MonitorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reset_hdf5_monitor_globals();
    output_dir_ = test_output_directory();
    std::filesystem::remove_all(output_dir_);
    jams::Jams::set_output_dir(output_dir_.string());
  }

  void TearDown() override {
    solver_.reset();
    reset_hdf5_monitor_globals();
    std::filesystem::remove_all(output_dir_);
  }

  void initialise(const std::string& monitor_config,
                  const bool use_regions = true) {
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(base_config(use_regions) + monitor_config);
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);

    if (globals::config->exists("initializer")) {
      jams::InitializerDispatcher::execute(globals::config->lookup("initializer"));
    }

    solver_ = std::make_unique<Hdf5MonitorStubSolver>();
    globals::solver = solver_.get();
    solver_->register_physics_module(Physics::create(globals::config->lookup("physics")));
    solver_->register_thermostat(Thermostat::create("classical-cpu", 1.0e-15));
  }

  void run_monitor_once() {
    {
      Hdf5Monitor monitor(first_monitor_settings());
      monitor.update(*solver_);
    }
  }

  const libconfig::Setting& first_monitor_settings() const {
    return globals::config->lookup("monitors")[0];
  }

  static std::vector<jams::Real> read_real_dataset(const std::string& path) {
    HighFive::File file(jams::output::monitor_filename("hdf5_lattice", "h5"),
                        HighFive::File::ReadOnly);
    std::vector<jams::Real> values;
    file.getDataSet(path).read(values);
    return values;
  }

  static std::string read_xdmf() {
    std::ifstream file(jams::output::monitor_filename("hdf5", "xdmf"));
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
  }

  static std::vector<size_t> lattice_dataset_dimensions(const std::string& path) {
    HighFive::File file(jams::output::monitor_filename("hdf5_lattice", "h5"),
                        HighFive::File::ReadOnly);
    return file.getDataSet(path).getDimensions();
  }

  static std::vector<size_t> time_dataset_dimensions(const std::string& path) {
    HighFive::File file(jams::output::monitor_filename_series("hdf5", "h5", 0),
                        HighFive::File::ReadOnly);
    return file.getDataSet(path).getDimensions();
  }

  static std::string base_config(const bool use_regions) {
    const std::string initializer = use_regions ? R"(
      initializer = {
        module = "damping-regions";
        regions = (
          {
            type = "constant";
            origin = [0.0, 0.0, 0.0];
            size = [2.0, 1.0, 1.0];
            alpha = 0.05;
          },
          {
            type = "linear";
            origin = [2.0, 0.0, 0.0];
            size = [2.0, 1.0, 1.0];
            direction = [1.0, 0.0, 0.0];
            low = 0.2;
            high = 0.4;
          }
        );
      };
    )" : "";

    const std::string thermostat = use_regions ? R"(
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
    )" : "";

    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        thermostat = "classical-cpu";
        t_step = 1.0e-15;
        t_min  = 1.0e-15;
        t_max  = 1.0e-15;
      };

      physics : {
        module = "empty";
        temperature = 42.0;
      };

      materials = (
        { name = "A"; moment = 1.5; alpha = 0.1; spin = [1.0, 0.0, 0.0]; }
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
        size = [4, 1, 1];
        periodic = [false, false, false];
        normalise_spins = false;
      };
    )") + initializer + thermostat;
  }

 private:
  static std::filesystem::path test_output_directory() {
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = "unknown";
    if (test_info != nullptr) {
      test_name = std::string(test_info->test_suite_name()) + "_" + test_info->name();
    }

    for (auto& ch : test_name) {
      const auto uch = static_cast<unsigned char>(ch);
      if (!std::isalnum(uch) && ch != '_' && ch != '-') {
        ch = '_';
      }
    }

    return std::filesystem::temp_directory_path() / ("jams_hdf5_monitor_test_" + test_name);
  }

  std::filesystem::path output_dir_;
  std::unique_ptr<Hdf5MonitorStubSolver> solver_;
};

TEST_F(Hdf5MonitorTest, WritesResolvedAlphaAndTemperatureWithNativeTypes) {
  initialise(R"(
    monitors = (
      { module = "hdf5"; output_steps = 1; compressed = false; }
    );
  )");

  run_monitor_once();

  const auto alpha = read_real_dataset("/alpha");
  ASSERT_EQ(alpha.size(), 4u);
  EXPECT_NEAR(static_cast<double>(alpha[0]), 0.05, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(alpha[1]), 0.05, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(alpha[2]), 0.2, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(alpha[3]), 0.3, 1.0e-6);

  const auto temperature = read_real_dataset("/temperature");
  ASSERT_EQ(temperature.size(), 4u);
  EXPECT_NEAR(static_cast<double>(temperature[0]), 100.0, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(temperature[1]), 100.0, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(temperature[2]), 200.0, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(temperature[3]), 300.0, 1.0e-6);

  HighFive::File file(jams::output::monitor_filename("hdf5_lattice", "h5"),
                      HighFive::File::ReadOnly);
  EXPECT_EQ(file.getDataSet("/alpha").getDataType().getClass(), HighFive::DataTypeClass::Float);
  EXPECT_EQ(file.getDataSet("/alpha").getDataType().getSize(), sizeof(jams::Real));
  EXPECT_EQ(file.getDataSet("/temperature").getDataType().getSize(), sizeof(jams::Real));
  EXPECT_EQ(file.getDataSet("/moments").getDataType().getSize(), sizeof(jams::Real));
  EXPECT_EQ(file.getDataSet("/positions").getDataType().getSize(), sizeof(double));

  const auto xdmf = read_xdmf();
  EXPECT_NE(xdmf.find("Attribute Name=\"Alpha\" AttributeType=\"Scalar\" Center=\"Node\""), std::string::npos);
  EXPECT_NE(xdmf.find("Attribute Name=\"Temperature\" AttributeType=\"Scalar\" Center=\"Node\""), std::string::npos);
  EXPECT_NE(xdmf.find("monitor_hdf5_lattice.h5:/alpha"), std::string::npos);
  EXPECT_NE(xdmf.find("monitor_hdf5_lattice.h5:/temperature"), std::string::npos);
  EXPECT_NE(xdmf.find("Precision=\"" + std::to_string(sizeof(jams::Real)) + "\""), std::string::npos);
}

TEST_F(Hdf5MonitorTest, UniformTemperatureWritesConstantTemperatureField) {
  initialise(R"(
    monitors = (
      { module = "hdf5"; output_steps = 1; compressed = false; }
    );
  )", false);

  run_monitor_once();

  const auto temperature = read_real_dataset("/temperature");
  ASSERT_EQ(temperature.size(), 4u);
  for (const auto value : temperature) {
    EXPECT_NEAR(static_cast<double>(value), 42.0, 1.0e-6);
  }
}

TEST_F(Hdf5MonitorTest, SliceOutputUsesConsistentFieldDimensions) {
  initialise(R"(
    monitors = (
      {
        module = "hdf5";
        output_steps = 1;
        compressed = false;
        slice = {
          origin = [0.5, -0.5, -0.5];
          size = [2.0, 1.0, 1.0];
        };
      }
    );
  )");

  run_monitor_once();

  EXPECT_EQ(lattice_dataset_dimensions("/positions"), (std::vector<size_t>{2, 3}));
  EXPECT_EQ(lattice_dataset_dimensions("/alpha"), (std::vector<size_t>{2}));
  EXPECT_EQ(lattice_dataset_dimensions("/temperature"), (std::vector<size_t>{2}));
  EXPECT_EQ(lattice_dataset_dimensions("/moments"), (std::vector<size_t>{2}));
  EXPECT_EQ(time_dataset_dimensions("/spins"), (std::vector<size_t>{2, 3}));

  const auto alpha = read_real_dataset("/alpha");
  ASSERT_EQ(alpha.size(), 2u);
  EXPECT_NEAR(static_cast<double>(alpha[0]), 0.05, 1.0e-6);
  EXPECT_NEAR(static_cast<double>(alpha[1]), 0.2, 1.0e-6);

  const auto xdmf = read_xdmf();
  EXPECT_NE(xdmf.find("Topology TopologyType=\"Polyvertex\" Dimensions=\"2\""), std::string::npos);
  EXPECT_NE(xdmf.find("DataItem Dimensions=\"2\" NumberType=\"Float\" Precision=\"" +
                      std::to_string(sizeof(jams::Real)) + "\" Format=\"HDF\""),
            std::string::npos);
}

}  // namespace jams::testing

#endif  // JAMS_TEST_MONITORS_TEST_HDF5_H
