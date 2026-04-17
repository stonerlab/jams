#ifndef JAMS_TEST_SOLVERS_TEST_LLG_DETERMINISTIC_DYNAMICS_H
#define JAMS_TEST_SOLVERS_TEST_LLG_DETERMINISTIC_DYNAMICS_H

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"

#if HAS_CUDA
#include <cuda_runtime.h>
#include "jams/cuda/cuda_solver.h"
#endif

namespace {

constexpr double kAlpha = 0.1;
constexpr double kFieldTesla = 100.0;
constexpr double kTimeStepPs = 0.001;  // 1 fs
constexpr double kTotalTimePs = 1.0;
constexpr int kSampleStride = 50;
constexpr double kMaxAngularErrorDeg = 0.05;
constexpr double kMaxNormError = 1e-6;
constexpr double kRadiansToDegrees = 57.2957795130823208768;

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size()
      && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string sanitize_test_name(const std::string& name) {
  auto sanitized = name;
  std::replace_if(
      sanitized.begin(),
      sanitized.end(),
      [](const char ch) { return !(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'); },
      '_');
  return sanitized;
}

std::string deterministic_single_spin_config(const std::string& solver_module) {
  std::ostringstream cfg;
  cfg << std::scientific << std::setprecision(17);
  cfg << R"(
materials = (
  {
    name = "A";
    moment = 1.0;
    alpha = )" << kAlpha << R"(;
    spin = [1.0, 0.0, 0.0];
  }
);

unitcell = {
  parameter = 3.0e-10;
  symops = false;
  basis = (
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  );
  positions = (
    ("A", [0.0, 0.0, 0.0])
  );
};

lattice = {
  size = [1, 1, 1];
  periodic = [false, false, false];
};

solver = {
  module = ")" << solver_module << R"(";
  t_step = )" << (kTimeStepPs * 1e-12) << R"(;
  t_max = )" << (kTotalTimePs * 1e-12) << R"(;
};

physics = {
  temperature = 0.0;
};

hamiltonians = (
  {
    module = "applied-field";
    field = [0.0, 0.0, )" << kFieldTesla << R"(];
  }
);
)";
  return cfg.str();
}

std::array<double, 3> analytic_single_spin_solution(const double time_ps,
                                                    const double gyro,
                                                    const double alpha,
                                                    const double field_tesla) {
  const double omega = gyro * field_tesla;
  const double tau = 1.0 / (alpha * gyro * field_tesla);
  return {
      std::cos(omega * time_ps) / std::cosh(time_ps / tau),
      std::sin(omega * time_ps) / std::cosh(time_ps / tau),
      std::tanh(time_ps / tau),
  };
}

double norm_error(const std::array<double, 3>& spin) {
  const double norm_sq = spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2];
  return std::abs(norm_sq - 1.0);
}

double angular_error_deg(const std::array<double, 3>& lhs,
                         const std::array<double, 3>& rhs) {
  const double dot = lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
  const double cross_x = lhs[1] * rhs[2] - lhs[2] * rhs[1];
  const double cross_y = lhs[2] * rhs[0] - lhs[0] * rhs[2];
  const double cross_z = lhs[0] * rhs[1] - lhs[1] * rhs[0];
  const double cross_norm = std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
  return std::atan2(cross_norm, dot) * kRadiansToDegrees;
}

void destroy_test_simulation() {
  delete globals::solver;
  globals::solver = nullptr;

  delete globals::lattice;
  globals::lattice = nullptr;

  globals::config.reset();
}

#if HAS_CUDA
bool have_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void synchronize_solver_state() {
  auto* cuda_solver = dynamic_cast<CudaSolver*>(globals::solver);
  if (cuda_solver != nullptr) {
    cuda_solver->synchronize_on_spin_barrier_event();
  }
}
#else
void synchronize_solver_state() {}
#endif

class DeterministicSingleSpinLLGTest : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    destroy_test_simulation();

#if HAS_CUDA
    if (ends_with(GetParam(), "-gpu") && !have_cuda_device()) {
      GTEST_SKIP() << "CUDA device not available";
    }
#endif

    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(deterministic_single_spin_config(GetParam()));

    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);

    globals::solver = Solver::create(globals::config->lookup("solver"));
    globals::solver->register_physics_module(Physics::create(globals::config->lookup("physics")));

    const auto& hamiltonian_settings = globals::config->lookup("hamiltonians");
    for (auto i = 0; i < hamiltonian_settings.getLength(); ++i) {
      globals::solver->register_hamiltonian(
          Hamiltonian::create(hamiltonian_settings[i], globals::num_spins, globals::solver->is_cuda_solver()));
    }
  }

  void TearDown() override {
    destroy_test_simulation();
  }
};

TEST_P(DeterministicSingleSpinLLGTest, FollowsAnalyticDampedTrajectory) {
  double max_angular_error_deg = 0.0;
  double max_spin_norm_error = 0.0;
  const int max_steps = globals::solver->max_steps();

  for (int step = 1; step <= max_steps; ++step) {
    globals::solver->update_physics_module();
    globals::solver->run();

    if (step % kSampleStride != 0 && step != max_steps) {
      continue;
    }

    synchronize_solver_state();

    const std::array<double, 3> actual = {
        globals::s(0, 0),
        globals::s(0, 1),
        globals::s(0, 2),
    };

    const auto expected = analytic_single_spin_solution(
        globals::solver->time(), globals::gyro(0), kAlpha, kFieldTesla);

    max_spin_norm_error = std::max(max_spin_norm_error, norm_error(actual));
    max_angular_error_deg = std::max(max_angular_error_deg, angular_error_deg(actual, expected));
  }

  EXPECT_LT(max_spin_norm_error, kMaxNormError);
  EXPECT_LT(max_angular_error_deg, kMaxAngularErrorDeg);
}

std::vector<std::string> deterministic_llg_solver_modules() {
  std::vector<std::string> solvers = {
      "llg-heun-cpu",
  };

#if HAS_CUDA
  solvers.emplace_back("llg-heun-gpu");
  solvers.emplace_back("llg-rk4-gpu");
  solvers.emplace_back("llg-rkmk2-gpu");
  solvers.emplace_back("llg-rkmk4-gpu");
  solvers.emplace_back("llg-simp-gpu");
#endif

  return solvers;
}

INSTANTIATE_TEST_SUITE_P(
    LLGSolvers,
    DeterministicSingleSpinLLGTest,
    ::testing::ValuesIn(deterministic_llg_solver_modules()),
    [](const ::testing::TestParamInfo<std::string>& info) {
      return sanitize_test_name(info.param);
    });

}  // namespace

#endif  // JAMS_TEST_SOLVERS_TEST_LLG_DETERMINISTIC_DYNAMICS_H
