#ifndef JAMS_TEST_INITIALIZER_TEST_H5_INITIALIZER_H
#define JAMS_TEST_INITIALIZER_TEST_H5_INITIALIZER_H

#include "gtest/gtest.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <libconfig.h++>

#include <jams/core/globals.h>
#include <jams/helpers/utils.h>
#include <jams/initializer/init_h5.h>
#include <jams/interface/highfive.h>

namespace {

void reset_h5_initializer_globals() {
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

class H5InitializerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reset_h5_initializer_globals();
    output_dir_ = test_output_directory();
    std::filesystem::remove_all(output_dir_);
    std::filesystem::create_directories(output_dir_);
    globals::config = std::make_unique<libconfig::Config>();
    resize_targets();
  }

  void TearDown() override {
    reset_h5_initializer_globals();
    std::filesystem::remove_all(output_dir_);
  }

  template <typename T>
  std::filesystem::path write_profile_file(
      const std::string& filename,
      const std::string& moment_dataset_path = "/mus",
      const bool zero_second_moment = false) const {
    const auto path = output_dir_ / filename;
    HighFive::File file(path.string(),
                        HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    jams::MultiArray<T, 2> spins(2, 3);
    spins(0, 0) = static_cast<T>(1.0);
    spins(0, 1) = static_cast<T>(0.0);
    spins(0, 2) = static_cast<T>(0.0);
    spins(1, 0) = static_cast<T>(0.0);
    spins(1, 1) = zero_second_moment ? static_cast<T>(0.0) : static_cast<T>(1.0);
    spins(1, 2) = static_cast<T>(0.0);

    jams::MultiArray<T, 1> alpha(2);
    alpha(0) = static_cast<T>(0.125);
    alpha(1) = static_cast<T>(0.25);

    jams::MultiArray<T, 1> mus(2);
    mus(0) = static_cast<T>(1.5);
    mus(1) = zero_second_moment ? static_cast<T>(0.0) : static_cast<T>(2.5);

    jams::MultiArray<T, 1> gyro(2);
    gyro(0) = static_cast<T>(3.5);
    gyro(1) = static_cast<T>(4.5);

    file.createDataSet<T>("/spins", HighFive::DataSpace({2, 3})).write(spins);
    file.createDataSet<T>("/alpha", HighFive::DataSpace({2})).write(alpha);
    file.createDataSet<T>(moment_dataset_path, HighFive::DataSpace({2})).write(mus);
    file.createDataSet<T>("/gyro", HighFive::DataSpace({2})).write(gyro);

    return path;
  }

  std::filesystem::path write_invalid_integer_alpha_file() const {
    const auto path = output_dir_ / "invalid_integer_alpha.h5";
    HighFive::File file(path.string(),
                        HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
    std::vector<int> alpha{1, 2};
    file.createDataSet<int>("/alpha", HighFive::DataSpace::From(alpha)).write(alpha);
    return path;
  }

  std::filesystem::path write_invalid_string_alpha_file() const {
    const auto path = output_dir_ / "invalid_string_alpha.h5";
    HighFive::File file(path.string(),
                        HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
    std::vector<std::string> alpha{"bad", "data"};
    file.createDataSet<std::string>("/alpha", HighFive::DataSpace::From(alpha)).write(alpha);
    return path;
  }

  void execute_initializer_for(const std::filesystem::path& path) {
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(std::string(R"(
      initializer = {
        module = "h5";
        spins = ")" + path.string() + R"(";
        alpha = ")" + path.string() + R"(";
        mus = ")" + path.string() + R"(";
        gyro = ")" + path.string() + R"(";
      };
    )"));

    jams::InitH5::execute(globals::config->lookup("initializer"));
  }

  void execute_alpha_initializer_for(const std::filesystem::path& path) {
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(std::string(R"(
      initializer = {
        module = "h5";
        alpha = ")" + path.string() + R"(";
      };
    )"));

    jams::InitH5::execute(globals::config->lookup("initializer"));
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

    return std::filesystem::temp_directory_path() / ("jams_h5_initializer_test_" + test_name);
  }

  static void resize_targets() {
    globals::num_spins = 2;
    globals::num_spins3 = 6;
    globals::s.resize(2, 3);
    globals::alpha.resize(2);
    globals::mus.resize(2);
    globals::gyro.resize(2);
  }

  std::filesystem::path output_dir_;
};

TEST_F(H5InitializerTest, LoadsFloatDatasetsIntoCurrentPrecision) {
  const auto path = write_profile_file<float>("float_profile.h5");

  execute_initializer_for(path);

  EXPECT_NEAR(globals::s(0, 0), 1.0, 1.0e-7);
  EXPECT_NEAR(globals::s(1, 1), 1.0, 1.0e-7);
  EXPECT_NEAR(static_cast<double>(globals::alpha(0)), 0.125, 1.0e-7);
  EXPECT_NEAR(static_cast<double>(globals::alpha(1)), 0.25, 1.0e-7);
  EXPECT_NEAR(static_cast<double>(globals::mus(0)), 1.5, 1.0e-7);
  EXPECT_EQ(globals::num_magnetic_spins, 2);
  EXPECT_NEAR(static_cast<double>(globals::inv_mus(0)), 1.0 / 1.5, 1.0e-7);
  EXPECT_NEAR(static_cast<double>(globals::gyro(1)), 4.5, 1.0e-7);
}

TEST_F(H5InitializerTest, LoadsDoubleDatasetsIntoCurrentPrecision) {
  const auto path = write_profile_file<double>("double_profile.h5");

  execute_initializer_for(path);

  EXPECT_NEAR(globals::s(0, 0), 1.0, 1.0e-12);
  EXPECT_NEAR(globals::s(1, 1), 1.0, 1.0e-12);
  EXPECT_NEAR(static_cast<double>(globals::alpha(0)), 0.125, 1.0e-7);
  EXPECT_NEAR(static_cast<double>(globals::alpha(1)), 0.25, 1.0e-7);
  EXPECT_NEAR(static_cast<double>(globals::mus(0)), 1.5, 1.0e-7);
  EXPECT_EQ(globals::num_magnetic_spins, 2);
  EXPECT_NEAR(static_cast<double>(globals::inv_mus(0)), 1.0 / 1.5, 1.0e-12);
  EXPECT_NEAR(static_cast<double>(globals::gyro(1)), 4.5, 1.0e-7);
}

TEST_F(H5InitializerTest, LoadsMonitorMomentDatasetAndRepairsZeroMomentSpin) {
  const auto path = write_profile_file<double>("monitor_profile.h5", "/moments", true);

  execute_initializer_for(path);

  EXPECT_NEAR(static_cast<double>(globals::mus(0)), 1.5, 1.0e-12);
  EXPECT_EQ(static_cast<double>(globals::mus(1)), 0.0);
  EXPECT_EQ(globals::num_magnetic_spins, 1);
  EXPECT_NEAR(static_cast<double>(globals::inv_mus(0)), 1.0 / 1.5, 1.0e-12);
  EXPECT_EQ(static_cast<double>(globals::inv_mus(1)), 0.0);
  EXPECT_EQ(globals::s(1, 0), 0.0);
  EXPECT_EQ(globals::s(1, 1), 0.0);
  EXPECT_EQ(globals::s(1, 2), 1.0);
}

TEST_F(H5InitializerTest, RejectsIntegerDatasetsForFloatingPointTargets) {
  const auto path = write_invalid_integer_alpha_file();

  EXPECT_THROW({
    execute_alpha_initializer_for(path);
  }, std::runtime_error);
}

TEST_F(H5InitializerTest, RejectsStringDatasetsForFloatingPointTargets) {
  const auto path = write_invalid_string_alpha_file();

  EXPECT_THROW({
    execute_alpha_initializer_for(path);
  }, std::runtime_error);
}

}  // namespace jams::testing

#endif  // JAMS_TEST_INITIALIZER_TEST_H5_INITIALIZER_H
