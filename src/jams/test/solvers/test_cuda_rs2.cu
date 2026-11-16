#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <type_traits>
#include <vector>

#include "jams/containers/vec3.h"
#include "jams/core/globals.h"
#include "jams/core/jams++.h"
#include "jams/cuda/cuda_common.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/exception.h"
#include "jams/solvers/cuda_llg_rs2.h"

class CudaRS2TestAccess {
 public:
  static void step_with_uniforms(CUDALLGRS2Solver& solver,
                                 const std::array<std::array<double, 5>, 2>& uniforms) {
    auto values = solver.uniforms_.mutable_host_view();
    for (int i = 0; i < 2; ++i) {
      for (int n = 0; n < 5; ++n) {
        values(i, n) = uniforms[i][n];
      }
    }
    solver.run_step();
  }

  static void install_coloured_bath(CUDALLGRS2Solver& solver, Thermostat* bath) {
    solver.register_thermostat(bath);
    solver.has_coloured_bath_ = true;
  }

  static const auto& uniforms(const CUDALLGRS2Solver& solver) { return solver.uniforms_; }
};

namespace {

using Spin = jams::Vec<double, 3>;
using SpinPair = std::array<Spin, 2>;
constexpr double kFieldTolerance = std::is_same_v<jams::Real, float> ? 2e-6 : 2e-12;

__host__ __device__ jams::Real coupled_field(const jams::Real neighbour, int component,
                                             double time) {
  return static_cast<jams::Real>(0.17 * neighbour + (component + 1) * (0.31 + time));
}

__global__ void record_trial_and_calculate_field(const jams::Real* spins, jams::Real* trial,
                                                jams::Real* fields, double time) {
  const int n = threadIdx.x;
  trial[n] = spins[n];
  const int component = n % 3;
  const int neighbour_offset = n < 3 ? 3 : 0;
  fields[n] = coupled_field(spins[neighbour_offset + (component + 1) % 3], component, time);
}

class RecordingHamiltonian final : public Hamiltonian {
 public:
  RecordingHamiltonian() : trials_(6, 2, 3) {
    set_name("rs2-test-coupled-field");
    field_.resize(2, 3);
    cuda_stream_ = CudaStream(CudaStream::Priority::LOW);
    CHECK_CUDA_STATUS(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
  }

  ~RecordingHamiltonian() override { cudaEventDestroy(done_); }

  void calculate_fields(double time, const SpinArray& spins) override {
    const auto trial_index = times.size();
    times.push_back(time);
    if (trial_index >= trials_.extent(0)) return;
    record_trial_and_calculate_field<<<1, 6, 0, get_stream()>>>(
        spins.device_data(), trials_.mutable_device_data() + 6 * trial_index,
        field_.mutable_device_data(), time);
    CHECK_CUDA_STATUS(cudaGetLastError());
  }

  jams::Vec<jams::Real, 3> calculate_field(int, double) override { return {}; }
  jams::Real calculate_energy(int, double) override { return 0.0; }

  std::vector<double> times;
  jams::MultiArray<jams::Real, 3> trials_;
};

__global__ void store_sample(jams::Real* noise, int sample) {
  noise[0] = 0.0;
  noise[1] = 0.0;
  noise[2] = static_cast<jams::Real>(sample);
}

class CountingThermostat final : public Thermostat {
 public:
  explicit CountingThermostat(double sample_step) : Thermostat(0.0, 0.0, sample_step, 1) {
    cuda_stream_ = CudaStream(CudaStream::Priority::LOW);
  }

  void update() override {
    store_sample<<<1, 1, 0, get_stream()>>>(noise_.mutable_device_data(), ++samples);
    CHECK_CUDA_STATUS(cudaGetLastError());
  }

  int samples = 0;
};

Spin spin_at(int i) {
  const auto spins = globals::s.host_view();
  return {spins(i, 0), spins(i, 1), spins(i, 2)};
}

Spin rotate(const Spin& u, const Spin& spin) {
  const double angle = std::sqrt(jams::dot(u, u));
  if (angle == 0.0) return spin;
  const auto axis = u / angle;
  const double cosine = std::cos(angle);
  return cosine * spin + std::sin(angle) * jams::cross(axis, spin)
      + (1.0 - cosine) * jams::dot(axis, spin) * axis;
}

Spin rotation_rate(const Spin& u, const Spin& spin, const Spin& field, int i) {
  const auto alpha = globals::alpha(i);
  const auto gyro = static_cast<jams::Real>(globals::gyro(i) / (1.0 + alpha * alpha));
  const auto omega = gyro * (field + alpha * jams::cross(spin, field));
  const auto first = jams::cross(u, omega);
  const double angle_squared = jams::dot(u, u);
  const double half_angle = 0.5 * std::sqrt(angle_squared);
  const double beta = angle_squared < 1e-6 ? 1.0 / 12.0 + angle_squared / 720.0
      : (1.0 - half_angle / std::tan(half_angle)) / angle_squared;
  return omega - 0.5 * first + beta * jams::cross(u, first);
}

SpinPair drift(const SpinPair& u, const SpinPair& spins, double time) {
  SpinPair rates;
  for (int i = 0; i < 2; ++i) {
    Spin field;
    for (int n = 0; n < 3; ++n) {
      const auto neighbour = static_cast<jams::Real>(spins[1 - i][(n + 1) % 3]);
      field[n] = coupled_field(neighbour, n, time) * globals::inv_mus(i);
    }
    rates[i] = rotation_rate(u[i], spins[i], field, i);
  }
  return rates;
}

struct ReferenceStep {
  std::array<SpinPair, 3> trials;
  SpinPair final_spins;
};

ReferenceStep reference_step(const SpinPair& initial, double time, double h) {
  constexpr std::array<Spin, 2> x_signs = {Spin{-1, 1, -1}, Spin{1, 0, 1}};
  constexpr std::array<Spin, 2> z_signs = {Spin{-1, 1, 0}, Spin{1, -1, 0}};
  constexpr std::array<double, 2> temperature = {30.0, 60.0};
  const auto sigma = [&](int i) {
    return std::sqrt(2.0 * kBoltzmannIU * globals::alpha(i) * globals::inv_mus(i)
                     * temperature[i] / globals::gyro(i));
  };
  const auto diffusion = [&](int i, int axis, const Spin& u) {
    Spin field;
    field[axis] = sigma(i);
    return rotation_rate(u, rotate(u, initial[i]), field, i);
  };

  ReferenceStep result;
  result.trials[0] = initial;
  const auto f1 = drift({}, initial, time);
  const double later_time = time + 2.0 * h / 3.0;
  const SpinPair u2 = {(2.0 * h / 3.0) * f1[0], (2.0 * h / 3.0) * f1[1]};
  for (int i = 0; i < 2; ++i) result.trials[1][i] = rotate(u2[i], initial[i]);
  const auto f2 = drift(u2, result.trials[1], later_time);

  SpinPair u3;
  SpinPair white;
  for (int i = 0; i < 2; ++i) {
    const Spin x = std::sqrt(3.0 * h) * x_signs[i];
    const Spin z = std::sqrt(h) * z_signs[i];
    std::array<Spin, 3> g1, g2;
    Spin b;
    for (int k = 0; k < 3; ++k) {
      g1[k] = diffusion(i, k, {});
      g2[k] = diffusion(i, k, (2.0 * x[k] / 3.0) * g1[k]);
      b = b + x[k] * (0.25 * g1[k] + 0.75 * g2[k]);
    }
    u3[i] = h * (f1[i] / 6.0 + 0.5 * f2[i]) + b;
    result.trials[2][i] = rotate(u3[i], initial[i]);

    std::array<std::array<double, 3>, 3> v{};
    for (int k = 1; k < 3; ++k) {
      for (int l = 0; l < k; ++l) {
        v[k][l] = x[k] * z[l];
        v[l][k] = -v[k][l];
      }
    }
    for (int k = 0; k < 3; ++k) {
      const auto g3 = diffusion(i, k, h * f1[i] + b - x[k] * (g1[k] / 6.0 + 0.5 * g2[k]));
      const auto g4 = diffusion(i, k, h * f1[i] + b + x[k] * (-1.5 * g1[k] - 0.5 * g2[k]
                                                            + 2.0 * g3));
      Spin pair;
      for (int l = 0; l < 3; ++l) pair = pair + (v[k][l] / std::sqrt(h)) * g1[l];
      white[i] = white[i] + (x[k] / 8.0) * (g1[k] + 3.0 * g2[k] + 3.0 * g3 + g4)
          + (std::sqrt(h) / 4.0) * (diffusion(i, k, -pair) - diffusion(i, k, pair));
    }
  }

  const auto f3 = drift(u3, result.trials[2], later_time);
  for (int i = 0; i < 2; ++i) {
    result.final_spins[i] = rotate(h * (0.25 * f1[i] + 0.25 * f2[i] + 0.5 * f3[i]) + white[i],
                                  initial[i]);
  }
  return result;
}

class CudaRS2Test : public testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
    old_output_ = jams::Jams::output_path();
    output_ = std::filesystem::temp_directory_path() /
              ("jams-cuda-rs2-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  }

  void TearDown() override {
    if (!initialized_) return;
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    jams::cleanup_simulation();
    globals::solver = nullptr;
    globals::lattice = nullptr;
    globals::config = nullptr;
    jams::Jams::set_mode(jams::Mode::CPU);
    jams::Jams::set_output_dir(old_output_);
    std::filesystem::remove_all(output_);
  }

  void initialize(const std::string& thermostat, double temperature, double alpha,
                  bool two_materials = false, const std::string& bath_settings = {}) {
    const std::string second_material = two_materials
        ? R"(, {name = "B"; moment = 2.1; alpha = 0.35; spin = [0.0, 0.0, 1.0];})" : "";
    const std::string second_site = two_materials ? R"(, ("B", [0.5, 0.0, 0.0]))" : "";
    const auto config = R"(
      sim = {seed = 1234567890;};
      materials = ({name = "A"; moment = 1.4; alpha = )" + std::to_string(alpha) + R"(;
                    spin = [1.0, 0.0, 0.0];})" + second_material + R"();
      unitcell = {parameter = 3e-10;
        basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        positions = (("A", [0.0, 0.0, 0.0]))" + second_site + R"();};
      lattice = {size = [1, 1, 1]; periodic = [false, false, false];};
      physics = {temperature = )" + std::to_string(temperature) + R"(;};
      hamiltonians = ();
      solver = {module = "llg-rs2-gpu"; thermostat = ")" + thermostat + R"(";
                gilbert_prefactor = true; t_step = 1e-14; t_max = 4e-14;};
      monitors = ();
    )" + bath_settings;

    jams::ProgramArgs args;
    args.output_path = output_.string();
    args.config_inputs.push_back({config, true});
    jams::initialize_simulation(args);
    initialized_ = true;
  }

  CUDALLGRS2Solver& solver() const { return static_cast<CUDALLGRS2Solver&>(*globals::solver); }

  void check_real_bath() {
    const auto initial = spin_at(0);
    bool field_is_nonzero = false;
    for (int step = 1; step <= 3; ++step) {
      solver().update_physics_module();
      solver().run();
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      for (int n = 0; n < 3; ++n) {
        const auto field = solver().thermostat()->field(0, n);
        ASSERT_TRUE(std::isfinite(field));
        field_is_nonzero |= field != 0.0;
      }
      const auto spin = spin_at(0);
      EXPECT_NEAR(jams::dot(spin, spin), 1.0, 2e-12);
      EXPECT_EQ(solver().iteration(), step);
      EXPECT_EQ(solver().time(), step * solver().time_step());
    }
    EXPECT_TRUE(field_is_nonzero);
    const auto change = spin_at(0) - initial;
    EXPECT_GT(jams::dot(change, change), 1e-16);
  }

 private:
  std::filesystem::path output_;
  std::string old_output_;
  bool initialized_ = false;
};

TEST_F(CudaRS2Test, FixedDrawsMatchCoupledTwoSpinReference) {
  initialize("classical-gpu", 30.0, 0.2, true, R"(
    thermostat = {default_temperature = 60.0;
      temperature_regions = ({type = "constant"; origin = [-0.1, -0.1, -0.1];
        size = [0.2, 0.2, 0.2]; temperature = 30.0;});};
  )");
  ASSERT_EQ(globals::num_spins, 2);
  auto* hamiltonian = new RecordingHamiltonian;
  solver().register_hamiltonian(hamiltonian);
  constexpr std::array<std::array<double, 5>, 2> uniforms = {{
      {0.08, 0.91, 0.10, 0.20, 0.80}, {0.92, 0.42, 0.89, 0.80, 0.25}}};
  const double h = solver().time_step();

  for (int step = 0; step < 2; ++step) {
    SCOPED_TRACE(step);
    if (step == 1) {
      // Emulate a between-step spin update after the accepted spin cache was published.
      auto spins = globals::s.mutable_host_view();
      spins(0, 0) = 0.0;
      spins(0, 1) = 1.0;
      spins(0, 2) = 0.0;
    }
    const SpinPair initial = {spin_at(0), spin_at(1)};
    const auto expected = reference_step(initial, solver().time(), h);
    solver().update_physics_module();
    CudaRS2TestAccess::step_with_uniforms(solver(), uniforms);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const int offset = 3 * step;
    ASSERT_EQ(hamiltonian->times.size(), offset + 3);
    EXPECT_EQ(hamiltonian->times[offset], step * h);
    EXPECT_DOUBLE_EQ(hamiltonian->times[offset + 1], step * h + 2.0 * h / 3.0);
    EXPECT_EQ(hamiltonian->times[offset + 2], hamiltonian->times[offset + 1]);
    const auto trials = hamiltonian->trials_.host_view();
    const auto final_cache = solver().spin_array_for_fields().host_view();
    for (int i = 0; i < 2; ++i) {
      const auto actual = spin_at(i);
      EXPECT_NEAR(jams::dot(actual, actual), jams::dot(initial[i], initial[i]), 2e-12);
      for (int n = 0; n < 3; ++n) {
        for (int stage = 0; stage < 3; ++stage) {
          EXPECT_NEAR(trials(offset + stage, i, n), expected.trials[stage][i][n], kFieldTolerance);
        }
        EXPECT_NEAR(actual[n], expected.final_spins[i][n], kFieldTolerance);
        EXPECT_EQ(final_cache(i, n), static_cast<jams::Real>(actual[n]));
      }
    }
    EXPECT_EQ(solver().iteration(), step + 1);
    EXPECT_EQ(solver().time(), (step + 1) * h);
  }
}

TEST_F(CudaRS2Test, ColouredStagesReuseLaterSampleAndAdvanceToEndpoint) {
  initialize("classical-gpu", 0.0, 0.0);
  const double h = solver().time_step();
  auto* bath = new CountingThermostat(h / 3.0);
  CudaRS2TestAccess::install_coloured_bath(solver(), bath);

  double angle = 0.0;
  for (int step = 0; step < 2; ++step) {
    solver().update_physics_module();
    solver().run();
    const int start_sample = 1 + 3 * step;
    const int later_sample = start_sample + 2;
    EXPECT_EQ(bath->samples, start_sample + 3);
    angle += h * globals::gyro(0) * (0.25 * start_sample + 0.75 * later_sample);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(bath->field(0, 2), 7.0);
  const auto spin = spin_at(0);
  EXPECT_NEAR(spin[0], std::cos(angle), 2e-12);
  EXPECT_NEAR(spin[1], std::sin(angle), 2e-12);
  EXPECT_EQ(spin[2], 0.0);
}

TEST_F(CudaRS2Test, WhiteBathGeneratesFiveUniformsForOneSpin) {
  initialize("classical-gpu", 50.0, 0.2);
  solver().update_physics_module();
  solver().run();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto uniforms = CudaRS2TestAccess::uniforms(solver()).host_span();
  ASSERT_EQ(uniforms.size(), 5u);
  for (const auto draw : uniforms) {
    EXPECT_GT(draw, 0.0);
    EXPECT_LE(draw, 1.0);
  }
  const auto spin = spin_at(0);
  EXPECT_NEAR(jams::dot(spin, spin), 1.0, 2e-12);
  EXPECT_EQ(solver().time(), solver().time_step());
}

TEST_F(CudaRS2Test, ChangingUniformWhiteTemperatureIsRejected) {
  class ChangingTemperaturePhysics final : public Physics {
   public:
    using Physics::Physics;
    void update(const int&, const double&, const double&) override { set_temperature(51.0); }
  };

  initialize("classical-gpu", 50.0, 0.2);
  solver().register_physics_module(new ChangingTemperaturePhysics(globals::config->lookup("physics")));
  solver().update_physics_module();
  EXPECT_THROW(solver().run(), jams::ConfigException);
  EXPECT_EQ(solver().iteration(), 0);
}

TEST_F(CudaRS2Test, QuantumSpdeBathSmoke) {
  initialize("quantum-spde-gpu", 50.0, 0.2);
  check_real_bath();
}

TEST_F(CudaRS2Test, GeneralFftBathSmoke) {
  initialize("general-fft-gpu", 50.0, 0.2, false,
             R"(thermostat = {spectrum = "classical-ohmic";};)");
  check_real_bath();
}

TEST_F(CudaRS2Test, GeneralFftZeroSpectrumKeepsSpinsFixed) {
  initialize("general-fft-gpu", 0.0, 0.2, false,
             R"(thermostat = {spectrum = "quantum-no-zero-ohmic";};)");
  const auto initial = spin_at(0);
  for (int step = 0; step < 2; ++step) {
    solver().update_physics_module();
    solver().run();
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto spin = spin_at(0);
  for (int n = 0; n < 3; ++n) {
    EXPECT_EQ(solver().thermostat()->field(0, n), 0.0);
    EXPECT_EQ(spin[n], initial[n]);
  }
}

}  // namespace
