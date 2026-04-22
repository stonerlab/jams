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

std::string deterministic_single_spin_descriptor_config(const std::string& backend,
                                                        const std::string& integrator,
                                                        const std::string& dynamics_body = "") {
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
  backend = ")" << backend << R"(";
  integrator = ")" << integrator << R"(";
  t_step = )" << (kTimeStepPs * 1e-12) << R"(;
  t_max = )" << (kTotalTimePs * 1e-12) << R"(;
};

dynamics = {
  equation = "llg";
)" << dynamics_body << R"(
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

std::string generic_sot_single_spin_config() {
  return deterministic_single_spin_descriptor_config(
      "cpu",
      "rk4",
      R"(
  terms = (
    {
      module = "sot";
      spin_polarisation = [0.0, 1.0, 0.0];
      spin_hall_angle = 0.2;
      charge_current_density = 5.0e11;
    }
  );
)");
}

std::string legacy_sot_single_spin_config() {
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
  module = "llg-sot-rk4-cpu";
  t_step = )" << (kTimeStepPs * 1e-12) << R"(;
  t_max = )" << (kTotalTimePs * 1e-12) << R"(;
  spin_polarisation = [0.0, 1.0, 0.0];
  spin_hall_angle = 0.2;
  charge_current_density = 5.0e11;
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

std::string surface_selector_config() {
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
  size = [1, 1, 3];
  periodic = [true, true, false];
};

solver = {
  backend = "cpu";
  integrator = "rk4";
  t_step = 5.0e-15;
  t_max = 2.5e-14;
};

dynamics = {
  equation = "llg";
  terms = (
    {
      module = "stt";
      coefficient = 2.0;
      spin_polarisation = [0.0, 0.0, 1.0];
      selector = {
        surface_layers = 1;
      };
    }
  );
};

physics = {
  temperature = 0.0;
};

hamiltonians = (
  {
    module = "applied-field";
    field = [0.0, 0.0, 0.0];
  }
);
)";
  return cfg.str();
}

std::string explicit_surface_site_config() {
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
  size = [1, 1, 3];
  periodic = [true, true, false];
};

solver = {
  backend = "cpu";
  integrator = "rk4";
  t_step = 5.0e-15;
  t_max = 2.5e-14;
};

dynamics = {
  equation = "llg";
  terms = (
    {
      module = "stt";
      coefficient = 2.0;
      spin_polarisation = [0.0, 0.0, 1.0];
      selector = {
        sites = [0, 2];
      };
    }
  );
};

physics = {
  temperature = 0.0;
};

hamiltonians = (
  {
    module = "applied-field";
    field = [0.0, 0.0, 0.0];
  }
);
)";
  return cfg.str();
}

std::string gse_cpu_descriptor_config() {
  return R"(
materials = (
  {
    name = "A";
    moment = 1.0;
    alpha = 0.1;
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
  backend = "cpu";
  integrator = "rk4";
  t_step = 1.0e-15;
  t_max = 1.0e-15;
};

dynamics = {
  equation = "gse";
};

physics = {
  temperature = 0.0;
};

hamiltonians = (
  {
    module = "applied-field";
    field = [0.0, 0.0, 0.0];
  }
);
)";
}

std::string ll_lorentzian_cpu_descriptor_config() {
  return R"(
materials = (
  {
    name = "A";
    moment = 1.0;
    alpha = 0.1;
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
  backend = "cpu";
  integrator = "rk4";
  t_step = 1.0e-15;
  t_max = 1.0e-15;
};

dynamics = {
  equation = "ll-lorentzian";
};

thermostat = {
  lorentzian_gamma = 1.0;
  lorentzian_omega0 = 1.0;
};

physics = {
  temperature = 0.0;
};

hamiltonians = (
  {
    module = "applied-field";
    field = [0.0, 0.0, 0.0];
  }
);
)";
}

std::string stochastic_cpu_single_spin_config(const std::string& thermostat_name,
                                              const std::string& thermostat_body = "") {
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
  backend = "cpu";
  integrator = "heun";
  thermostat = ")" << thermostat_name << R"(";
  t_step = )" << (kTimeStepPs * 1e-12) << R"(;
  t_max = )" << (kTimeStepPs * 1e-12) << R"(;
};

dynamics = {
  equation = "llg";
};

physics = {
  temperature = 300.0;
};

thermostat = {
)" << thermostat_body << R"(
};

hamiltonians = (
  {
    module = "applied-field";
    field = [0.0, 0.0, 0.0];
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

void setup_test_simulation(const std::string& config_text) {
  destroy_test_simulation();
  globals::config = std::make_unique<libconfig::Config>();
  globals::config->readString(config_text);

  globals::lattice = new Lattice();
  globals::lattice->init_from_config(*globals::config);

  globals::solver = Solver::create(globals::config->lookup("solver"));
  globals::solver->register_physics_module(Physics::create(globals::config->lookup("physics")));

  if (globals::config->exists("hamiltonians")) {
    const auto& hamiltonian_settings = globals::config->lookup("hamiltonians");
    for (auto i = 0; i < hamiltonian_settings.getLength(); ++i) {
      globals::solver->register_hamiltonian(
          Hamiltonian::create(hamiltonian_settings[i], globals::num_spins, globals::solver->is_cuda_solver()));
    }
  }
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

std::vector<std::array<double, 3>> run_config_to_completion(const std::string& config_text) {
  setup_test_simulation(config_text);

  for (int step = 1; step <= globals::solver->max_steps(); ++step) {
    globals::solver->update_physics_module();
    globals::solver->run();
  }

  synchronize_solver_state();

  std::vector<std::array<double, 3>> spins(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    spins[i] = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
  }

  destroy_test_simulation();
  return spins;
}

class DeterministicSingleSpinLLGTest : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    destroy_test_simulation();

#if HAS_CUDA
    if (ends_with(GetParam(), "-gpu") && !have_cuda_device()) {
      GTEST_SKIP() << "CUDA device not available";
    }
#endif

    setup_test_simulation(deterministic_single_spin_config(GetParam()));
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
      "llg-rk4-cpu",
      "llg-rkmk2-cpu",
      "llg-rkmk4-cpu",
      "llg-simp-cpu",
      "llg-dm-cpu",
  };

#if HAS_CUDA
  solvers.emplace_back("llg-heun-gpu");
  solvers.emplace_back("llg-rk4-gpu");
  solvers.emplace_back("llg-rkmk2-gpu");
  solvers.emplace_back("llg-rkmk4-gpu");
  solvers.emplace_back("llg-simp-gpu");
  solvers.emplace_back("llg-dm-gpu");
#endif

  return solvers;
}

TEST(SolverDescriptorConfigTest, BackendIntegratorEquationConfigMatchesLegacyModule) {
  const auto legacy_spin = run_config_to_completion(deterministic_single_spin_config("llg-rk4-cpu")).front();
  const auto generic_spin = run_config_to_completion(
      deterministic_single_spin_descriptor_config("cpu", "rk4")).front();

  EXPECT_LT(angular_error_deg(legacy_spin, generic_spin), 1e-10);
  EXPECT_LT(norm_error(generic_spin), kMaxNormError);
}

TEST(LLGDynamicsTermConfigTest, GenericSotTermMatchesLegacySotAlias) {
  const auto legacy_spin = run_config_to_completion(legacy_sot_single_spin_config()).front();
  const auto generic_spin = run_config_to_completion(generic_sot_single_spin_config()).front();

  EXPECT_LT(angular_error_deg(legacy_spin, generic_spin), 1e-10);
  EXPECT_LT(norm_error(generic_spin), kMaxNormError);
}

TEST(LLGDynamicsSelectorTest, SurfaceSelectorMatchesExplicitSurfaceSites) {
  const auto surface_spins = run_config_to_completion(surface_selector_config());
  const auto explicit_spins = run_config_to_completion(explicit_surface_site_config());

  ASSERT_EQ(surface_spins.size(), explicit_spins.size());
  for (std::size_t i = 0; i < surface_spins.size(); ++i) {
    EXPECT_LT(angular_error_deg(surface_spins[i], explicit_spins[i]), 1e-10);
    EXPECT_LT(norm_error(surface_spins[i]), kMaxNormError);
  }
}

TEST(AdditionalCpuBackendsSmokeTest, GseAndLorentzianCpuBackendsRunOneStep) {
  const auto gse_spin = run_config_to_completion(gse_cpu_descriptor_config()).front();
  const auto lorentzian_spin = run_config_to_completion(ll_lorentzian_cpu_descriptor_config()).front();
  const std::array<double, 3> expected = {1.0, 0.0, 0.0};

  EXPECT_LT(angular_error_deg(gse_spin, expected), 1e-10);
  EXPECT_LT(angular_error_deg(lorentzian_spin, expected), 1e-10);
  EXPECT_LT(norm_error(gse_spin), kMaxNormError);
  EXPECT_LT(norm_error(lorentzian_spin), kMaxNormError);
}

TEST(CpuThermostatIntegrationSmokeTest, QuantumSpdeAndGeneralFftRunWithCpuHeunSolver) {
  jams::instance().random_generator().seed(0x1234ULL);
  const auto quantum_spin = run_config_to_completion(
      stochastic_cpu_single_spin_config("quantum-spde-cpu", "zero_point = false;")).front();

  jams::instance().random_generator().seed(0x5678ULL);
  const auto fft_spin = run_config_to_completion(
      stochastic_cpu_single_spin_config(
          "general-fft-cpu",
          R"(
  spectrum = "classical";
  write_diagnostics = false;
)")).front();

  for (const auto spin : {quantum_spin, fft_spin}) {
    EXPECT_TRUE(std::isfinite(spin[0]));
    EXPECT_TRUE(std::isfinite(spin[1]));
    EXPECT_TRUE(std::isfinite(spin[2]));
    EXPECT_LT(norm_error(spin), kMaxNormError);
  }
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
