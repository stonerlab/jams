#ifndef JAMS_TEST_MONITORS_TEST_CUDA_THERMAL_CURRENT_H
#define JAMS_TEST_MONITORS_TEST_CUDA_THERMAL_CURRENT_H

#include "gtest/gtest.h"

#include <filesystem>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <libconfig.h++>

#include <jams/common.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/hamiltonian/cuda_dipole_fft.h>
#include <jams/helpers/output.h>
#include <jams/helpers/utils.h>
#include <jams/monitors/cuda_thermal_current.h>
#include <jams/test/hamiltonian/test_dipole_input.h>

namespace {

bool thermal_current_cuda_device_is_available() {
  int device_count = 0;
  const auto device_status = cudaGetDeviceCount(&device_count);
  return device_status == cudaSuccess && device_count > 0;
}

class CudaThermalCurrentStubSolver : public Solver {
public:
  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "cuda-thermal-current-stub"; }
  bool is_cuda_solver() const override { return true; }
};

class CudaThermalCurrentMonitorTest : public ::testing::Test {
protected:
  void SetUp() override {
    globals::solver = nullptr;
    output_dir_ = std::filesystem::temp_directory_path()
        / "jams_cuda_thermal_current_monitor_test";
    std::filesystem::remove_all(output_dir_);
    jams::Jams::set_output_dir(output_dir_.string());
  }

  void TearDown() override {
    globals::solver = nullptr;
    jams::Jams::set_mode(jams::Mode::CPU);
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

    if (globals::lattice) {
      delete globals::lattice;
      globals::lattice = nullptr;
    }

    std::filesystem::remove_all(output_dir_);
  }

  void initialise(const std::string& monitor_settings) {
    using namespace jams::testing::dipole;

    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(
        config_basic_gpu
        + config_unitcell_sc
        + config_lattice({3.0, 3.0, 3.0}, {true, true, true})
        + config_dipole("dipole-fft", 1.1)
        + monitor_settings);
    globals::lattice->init_from_config(*globals::config);
  }

  const libconfig::Setting& hamiltonian_settings() const {
    return globals::config->lookup("hamiltonians")[0];
  }

  const libconfig::Setting& monitor_settings() const {
    return globals::config->lookup("monitors")[0];
  }

  std::filesystem::path output_dir_;
};

TEST_F(CudaThermalCurrentMonitorTest, IncludesDipoleFftProviderByDefault) {
  if (!thermal_current_cuda_device_is_available()) {
    GTEST_SKIP() << "CUDA device is not available";
  }
  cudaDeviceReset();
  jams::Jams::set_mode(jams::Mode::GPU);

  initialise(R"(
    monitors = (
      { module = "thermal-current"; output_steps = 1; }
    );
  )");

  CudaThermalCurrentStubSolver solver;
  globals::solver = &solver;

  auto* dipole_fft = new CudaDipoleFFTHamiltonian(
      hamiltonian_settings(),
      globals::num_spins);
  ASSERT_EQ(dipole_fft->energy_current_tensor_memory(), 0u);
  solver.register_hamiltonian(dipole_fft);

  CudaThermalCurrentMonitor monitor(monitor_settings());
  EXPECT_GT(dipole_fft->energy_current_tensor_memory(), 0u);

  globals::solver = nullptr;
}

TEST_F(CudaThermalCurrentMonitorTest, ExcludesDipoleFftCaseInsensitively) {
  if (!thermal_current_cuda_device_is_available()) {
    GTEST_SKIP() << "CUDA device is not available";
  }
  cudaDeviceReset();
  jams::Jams::set_mode(jams::Mode::GPU);

  initialise(R"(
    monitors = (
      {
        module = "thermal-current";
        output_steps = 1;
        exclude_hamiltonians = ["DIPOLE-FFT"];
      }
    );
  )");

  CudaThermalCurrentStubSolver solver;
  globals::solver = &solver;

  auto* dipole_fft = new CudaDipoleFFTHamiltonian(
      hamiltonian_settings(),
      globals::num_spins);
  ASSERT_EQ(dipole_fft->energy_current_tensor_memory(), 0u);
  solver.register_hamiltonian(dipole_fft);

  CudaThermalCurrentMonitor monitor(monitor_settings());
  EXPECT_EQ(dipole_fft->energy_current_tensor_memory(), 0u);

  globals::solver = nullptr;
}

}  // namespace

#endif  // JAMS_TEST_MONITORS_TEST_CUDA_THERMAL_CURRENT_H
