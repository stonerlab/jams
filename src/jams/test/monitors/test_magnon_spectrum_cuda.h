#ifndef JAMS_TEST_MONITORS_TEST_MAGNON_SPECTRUM_CUDA_H
#define JAMS_TEST_MONITORS_TEST_MAGNON_SPECTRUM_CUDA_H

#include "gtest/gtest.h"

#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <libconfig.h++>

#include <jams/common.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/consts.h>
#include <jams/helpers/output.h>
#include <jams/helpers/utils.h>
#include <jams/interface/fft.h>
#include <jams/monitors/magnon_density.h>
#include <jams/monitors/magnon_spectrum.h>
#include <jams/monitors/spectrum_base.h>

#if HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace jams::testing {

inline void reset_magnon_spectrum_monitor_globals() {
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
  globals::solver = nullptr;
  globals::config = nullptr;
  if (globals::lattice != nullptr) {
    delete globals::lattice;
    globals::lattice = nullptr;
  }
}

class MagnonSpectrumStubSolver : public Solver {
public:
  MagnonSpectrumStubSolver() { step_size_ = 0.25; }

  void initialize(const libconfig::Setting&) override {}
  void run() override {}
  std::string name() const override { return "magnon-spectrum-stub"; }
};

class CartesianSpectrumProbeMonitor final : public SpectrumBaseMonitor {
public:
  using SpectrumRows = std::vector<std::vector<double>>;

  explicit CartesianSpectrumProbeMonitor(const libconfig::Setting& settings)
      : SpectrumBaseMonitor(settings) {
    enable_cuda_frequency_slices_backend_();
    validate_cuda_time_fft_backend_support_();
  }

  void post_process() override {}

  void update(Solver& solver) override {
    store_sk_snapshot(globals::s);
    if (!periodogram_window_complete()) {
      return;
    }

    const auto& spectrum = finalise_periodogram_spectrum();
    rows_.clear();
    for (std::size_t a = 0; a < spectrum.extent(0); ++a) {
      for (std::size_t f = 0; f < spectrum.extent(1); ++f) {
        for (std::size_t k = 0; k < spectrum.extent(2); ++k) {
          for (std::size_t c = 0; c < spectrum.extent(3); ++c) {
            const auto value = spectrum(a, f, k, c);
            rows_.push_back({value.real(), value.imag()});
          }
        }
      }
    }
  }

  const SpectrumRows& rows() const { return rows_; }

private:
  SpectrumRows rows_;
};

class RaisedSpectrumProbeMonitor final : public SpectrumBaseMonitor {
public:
  struct PowerRow {
    int basis = 0;
    int frequency = 0;
    int k = 0;
    double power = 0.0;
  };

  explicit RaisedSpectrumProbeMonitor(const libconfig::Setting& settings)
      : SpectrumBaseMonitor(settings, KSamplingMode::FullGrid) {
    require_negative_frequencies_();
    auto channel_map = raise_lower_channel_map();
    channel_map.output_channels = 1;
    set_channel_map(channel_map);
  }

  void post_process() override {}

  void update(Solver& solver) override {
    store_sk_snapshot(globals::s);
    if (!periodogram_window_complete()) {
      return;
    }

    const auto& spectrum = finalise_periodogram_spectrum();
    powers_.clear();
    for (std::size_t a = 0; a < spectrum.extent(0); ++a) {
      for (std::size_t f = 0; f < spectrum.extent(1); ++f) {
        for (std::size_t k = 0; k < spectrum.extent(2); ++k) {
          powers_.push_back({
              static_cast<int>(a),
              static_cast<int>(f),
              static_cast<int>(k),
              std::norm(spectrum(a, f, k, 0))});
        }
      }
    }
  }

  double spin_length_for_test(const int basis) const {
    return basis_spin_length_(basis);
  }

  const std::vector<PowerRow>& powers() const { return powers_; }

private:
  std::vector<PowerRow> powers_;
};

#if HAS_CUDA
class MagnonSpectrumCudaStubSolver : public MagnonSpectrumStubSolver {
public:
  bool is_cuda_solver() const override { return true; }
};

inline bool magnon_spectrum_cuda_device_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

class MagnonSpectrumCudaMonitorTest : public ::testing::Test {
protected:
  using SpectrumTable = std::vector<std::vector<double>>;
  using DensityTable = std::vector<std::vector<double>>;

  struct RaisedProbeResult {
    std::vector<RaisedSpectrumProbeMonitor::PowerRow> powers;
    std::vector<double> spin_lengths;
  };

  void SetUp() override {
    reset_magnon_spectrum_monitor_globals();
    output_dir_ = std::filesystem::temp_directory_path()
        / ("jams_magnon_spectrum_monitor_test_" + current_test_id());
    std::filesystem::remove_all(output_dir_);
  }

  void TearDown() override {
    reset_magnon_spectrum_monitor_globals();
    std::filesystem::remove_all(output_dir_);
  }

  SpectrumTable run_spectrum(
      Solver& solver,
      const std::string& run_name,
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib = 0) {
    initialise_lattice(estimator, spatial_backend, time_backend, cuda_memory_limit_mib);

    const auto run_dir = output_dir_ / run_name;
    std::filesystem::remove_all(run_dir);
    std::filesystem::create_directories(run_dir);
    jams::Jams::set_output_dir(run_dir.string());

    globals::solver = &solver;
    {
      MagnonSpectrumMonitor monitor(first_monitor_settings());
      for (int t = 0; t < periodogram_length_; ++t) {
        write_spin_state(t);
        monitor.update(solver);
      }
    }
    globals::solver = nullptr;

    return read_spectrum_rows();
  }

  DensityTable run_density(
      Solver& solver,
      const std::string& run_name,
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib = 0,
      const std::string& circular_channels = "") {
    initialise_density_lattice(estimator, spatial_backend, time_backend, cuda_memory_limit_mib, circular_channels);

    const auto run_dir = output_dir_ / run_name;
    std::filesystem::remove_all(run_dir);
    std::filesystem::create_directories(run_dir);
    jams::Jams::set_output_dir(run_dir.string());

    globals::solver = &solver;
    {
      MagnonDensityMonitor monitor(first_monitor_settings());
      for (int t = 0; t < periodogram_length_; ++t) {
        write_spin_state(t);
        monitor.update(solver);
      }
    }
    globals::solver = nullptr;

    return read_density_rows();
  }

  DensityTable run_one_basis_density(
      Solver& solver,
      const std::string& run_name,
      const std::string& circular_channels = "") {
    initialise_one_basis_density_lattice(circular_channels);

    const auto run_dir = output_dir_ / run_name;
    std::filesystem::remove_all(run_dir);
    std::filesystem::create_directories(run_dir);
    jams::Jams::set_output_dir(run_dir.string());

    globals::solver = &solver;
    {
      MagnonDensityMonitor monitor(first_monitor_settings());
      for (int t = 0; t < periodogram_length_; ++t) {
        write_one_basis_known_spin_state(t);
        monitor.update(solver);
      }
    }
    globals::solver = nullptr;

    return read_density_rows();
  }

  RaisedProbeResult run_raised_probe(
      Solver& solver,
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib = 0) {
    initialise_density_lattice(estimator, spatial_backend, time_backend, cuda_memory_limit_mib);

    globals::solver = &solver;
    RaisedProbeResult result;
    {
      RaisedSpectrumProbeMonitor monitor(first_monitor_settings());
      for (int t = 0; t < periodogram_length_; ++t) {
        write_spin_state(t);
        monitor.update(solver);
      }
      result.powers = monitor.powers();
      result.spin_lengths.reserve(static_cast<std::size_t>(globals::lattice->num_basis_sites()));
      for (int a = 0; a < globals::lattice->num_basis_sites(); ++a) {
        result.spin_lengths.push_back(monitor.spin_length_for_test(a));
      }
    }
    globals::solver = nullptr;
    return result;
  }

  SpectrumTable run_cartesian_probe(
      Solver& solver,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const std::string& estimator = "welch") {
    initialise_cartesian_lattice(spatial_backend, time_backend, estimator);

    globals::solver = &solver;
    SpectrumTable rows;
    {
      CartesianSpectrumProbeMonitor monitor(first_monitor_settings());
      for (int t = 0; t < periodogram_length_; ++t) {
        write_spin_state(t);
        monitor.update(solver);
      }
      rows = monitor.rows();
    }
    globals::solver = nullptr;
    return rows;
  }

  void initialise_lattice(
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib = 0) {
    delete globals::lattice;
    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(base_config(
        estimator,
        spatial_backend,
        time_backend,
        cuda_memory_limit_mib));
    globals::lattice->init_from_config(*globals::config);
  }

  void initialise_density_lattice(
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib = 0,
      const std::string& circular_channels = "") {
    delete globals::lattice;
    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(density_config(
        estimator,
        spatial_backend,
        time_backend,
        cuda_memory_limit_mib,
        circular_channels));
    globals::lattice->init_from_config(*globals::config);
  }

  void initialise_one_basis_density_lattice(const std::string& circular_channels = "") {
    delete globals::lattice;
    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(one_basis_density_config(circular_channels));
    globals::lattice->init_from_config(*globals::config);
  }

  void initialise_cartesian_lattice(
      const std::string& spatial_backend,
      const std::string& time_backend,
      const std::string& estimator = "welch") {
    delete globals::lattice;
    globals::lattice = new Lattice();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(cartesian_probe_config(spatial_backend, time_backend, estimator));
    globals::lattice->init_from_config(*globals::config);
  }

  const libconfig::Setting& first_monitor_settings() const {
    return globals::config->lookup("monitors")[0];
  }

  void write_spin_state(const int time_index) {
    auto spins = globals::s.mutable_host_view();
    for (int spin = 0; spin < globals::num_spins; ++spin) {
      const double site = static_cast<double>(spin);
      const double t = static_cast<double>(time_index);
      const double phase = 0.77 * t + 0.31 * site;
      spins(spin, 0) = 0.07 * std::cos(phase) + 0.015 * std::sin(0.37 * t + 0.11 * site);
      spins(spin, 1) = 0.05 * std::sin(1.19 * phase + 0.23 * site);
      spins(spin, 2) = 1.0 + 0.02 * std::cos(0.41 * t + 0.17 * site);
    }
  }

  void write_one_basis_known_spin_state(const int time_index) {
    auto spins = globals::s.mutable_host_view();
    ASSERT_EQ(globals::num_spins, 1);
    spins(0, 0) = one_basis_known_x(time_index);
    spins(0, 1) = one_basis_known_y(time_index);
    spins(0, 2) = 1.0;
  }

  static SpectrumTable read_spectrum_rows() {
    std::ifstream file(jams::output::monitor_filename_series("magnon-spectrum_path", "tsv", 0, 1));
    EXPECT_TRUE(file.good());

    SpectrumTable rows;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream stream(line);
      std::vector<double> values;
      double value = 0.0;
      while (stream >> value) {
        values.push_back(value);
      }
      if (!values.empty()) {
        rows.push_back(std::move(values));
      }
    }
    return rows;
  }

  static DensityTable read_density_rows() {
    std::ifstream file(jams::output::monitor_filename("magnon-density", "tsv"));
    EXPECT_TRUE(file.good());

    DensityTable rows;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream stream(line);
      std::vector<double> values;
      double value = 0.0;
      while (stream >> value) {
        values.push_back(value);
      }
      if (!values.empty()) {
        rows.push_back(std::move(values));
      }
    }
    return rows;
  }

  static std::vector<std::string> read_density_header() {
    std::ifstream file(jams::output::monitor_filename("magnon-density", "tsv"));
    EXPECT_TRUE(file.good());

    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream stream(line);
      std::vector<std::string> columns;
      std::string column;
      while (stream >> column) {
        columns.push_back(std::move(column));
      }
      return columns;
    }
    return {};
  }

  static void expect_spectra_near(
      const SpectrumTable& actual,
      const SpectrumTable& expected,
      const double relative_tolerance = 1.0e-5,
      const double absolute_tolerance = 1.0e-8) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t row = 0; row < expected.size(); ++row) {
      ASSERT_EQ(actual[row].size(), expected[row].size()) << "row " << row;
      for (std::size_t col = 0; col < expected[row].size(); ++col) {
        const double scale = std::max(std::abs(expected[row][col]), 1.0);
        EXPECT_NEAR(actual[row][col], expected[row][col], absolute_tolerance + relative_tolerance * scale)
            << "row " << row << " col " << col;
      }
    }
  }

  static void expect_finite_spectrum(const SpectrumTable& rows) {
    ASSERT_FALSE(rows.empty());
    for (const auto& row : rows) {
      ASSERT_EQ(row.size(), 13u);
      for (const double value : row) {
        EXPECT_TRUE(std::isfinite(value));
      }
    }
  }

  static void expect_finite_density(const DensityTable& rows, const std::size_t expected_columns = 4) {
    ASSERT_FALSE(rows.empty());
    for (const auto& row : rows) {
      ASSERT_EQ(row.size(), expected_columns);
      for (const double value : row) {
        EXPECT_TRUE(std::isfinite(value));
      }
    }
  }

  static void expect_density_header_equals(const std::vector<std::string>& actual,
                                           const std::vector<std::string>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(actual[i], expected[i]) << "column " << i;
    }
  }

  static void expect_density_folded_columns(
      const DensityTable& rows,
      const int two_sided_column,
      const int folded_column,
      const double df) {
    for (const auto& row : rows) {
      const int freq_bin = static_cast<int>(std::llround(row[0] / df));
      if (freq_bin < 0) {
        EXPECT_NEAR(row[folded_column], 0.0, 1.0e-12);
        continue;
      }

      if (freq_bin == 0 || (periodogram_length_ % 2 == 0 && freq_bin == periodogram_length_ / 2)) {
        EXPECT_NEAR(
            row[folded_column],
            row[two_sided_column],
            std::max(1.0, std::abs(row[two_sided_column])) * 1.0e-8);
        continue;
      }

      const auto negative = std::find_if(
          rows.begin(),
          rows.end(),
          [&](const auto& candidate) {
            return static_cast<int>(std::llround(candidate[0] / df)) == -freq_bin;
          });
      ASSERT_NE(negative, rows.end());
      const double expected_folded = row[two_sided_column] + (*negative)[two_sided_column];
      EXPECT_NEAR(
          row[folded_column],
          expected_folded,
          std::max(1.0, std::abs(expected_folded)) * 1.0e-8);
    }
  }

  static std::string current_test_id() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string id = info
        ? std::string(info->test_suite_name()) + "_" + info->name()
        : "unknown";
    for (char& ch : id) {
      const auto uch = static_cast<unsigned char>(ch);
      if (!std::isalnum(uch) && ch != '_' && ch != '-') {
        ch = '_';
      }
    }
    return id;
  }

  static std::string periodogram_config(const std::string& estimator) {
    if (estimator == "welch") {
      return R"(
          estimator = "welch";
      )";
    }

    if (estimator == "multitaper") {
      return R"(
          estimator = "multitaper";
          multitaper_bandwidth = 2.0;
          multitaper_tapers = 2;
      )";
    }

    throw std::runtime_error("unexpected estimator in test");
  }

  static double one_basis_known_x(const int time_index) {
    return one_basis_transverse_scale_ * (static_cast<double>(time_index) - 0.5 * (periodogram_length_ - 1));
  }

  static double one_basis_known_y(const int time_index) {
    const double centre = static_cast<double>(time_index) - 0.5 * (periodogram_length_ - 1);
    const double half_width = 0.5 * (periodogram_length_ - 1);
    return one_basis_transverse_scale_ * 0.35 * centre * centre * centre / (half_width * half_width);
  }

  static std::vector<double> expected_one_basis_density_two_sided(
      const double dt_ps,
      const int circular_channel) {
    const int N = periodogram_length_;
    const double spin_length =
        (one_basis_moment_mu_b_ * kBohrMagnetonIU) / (kElectronGFactor * kBohrMagnetonIU);

    std::vector<std::complex<double>> circular_samples(static_cast<std::size_t>(N));
    std::complex<double> mean = 0.0;
    for (int t = 0; t < N; ++t) {
      const double y = (circular_channel == 0) ? one_basis_known_y(t) : -one_basis_known_y(t);
      const std::complex<double> sample =
          spin_length * kInvSqrtTwo * std::complex<double>(one_basis_known_x(t), y);
      circular_samples[static_cast<std::size_t>(t)] = {
          static_cast<float>(sample.real()),
          static_cast<float>(sample.imag())};
      mean += circular_samples[static_cast<std::size_t>(t)];
    }
    mean /= static_cast<double>(N);

    std::vector<double> window(static_cast<std::size_t>(N));
    double w2sum = 0.0;
    for (int t = 0; t < N; ++t) {
      const double w = fft_window_default(t, N);
      window[static_cast<std::size_t>(t)] = w;
      w2sum += w * w;
    }
    const double inv_rms = 1.0 / std::sqrt(w2sum / static_cast<double>(N));
    for (double& w : window) {
      w *= inv_rms;
    }

    const double df = 1.0 / (static_cast<double>(N) * dt_ps);
    const double volume = pow3(one_basis_parameter_m_);
    const double prefactor = 1.0 / (volume * df * kTHz2meV);
    std::vector<double> expected(static_cast<std::size_t>(N), 0.0);
    for (int f = 0; f < N; ++f) {
      std::complex<double> spectrum = 0.0;
      for (int t = 0; t < N; ++t) {
        const auto centred = (circular_samples[static_cast<std::size_t>(t)] - mean) / static_cast<double>(N);
        const auto windowed = window[static_cast<std::size_t>(t)] * centred;
        const double phase = -kTwoPi * static_cast<double>(f * t) / static_cast<double>(N);
        spectrum += windowed * std::exp(std::complex<double>(0.0, phase));
      }
      expected[static_cast<std::size_t>(f)] = prefactor * std::norm(spectrum) / spin_length;
    }
    return expected;
  }

  static std::string base_config(
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib) {
    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 2.5e-13;
        t_min  = 2.5e-13;
        t_max  = 2.0e-12;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; },
        { name = "B"; moment = 1.5; spin = [0.0, 0.0, 1.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.00, 0.0, 0.0]),
          ("B", [0.50, 0.0, 0.0])
        );
      };

      lattice : {
        size = [2, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };

      monitors = (
        {
          module = "magnon-spectrum";
          output_steps = 1;
          output_magnon_spectrum = true;
          site_resolved = false;
          keep_negative_frequencies = false;
          fftw_threads = 1;
          sk_time_series_backend = "memory";
          spatial_fft_backend = ")") + spatial_backend + R"(";
          time_fft_backend = ")" + time_backend + R"(";
          cuda_time_fft_memory_limit_mib = )" + std::to_string(cuda_memory_limit_mib) + R"(;
          hkl_path = (
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]
          );
          compute_periodogram : {
            length = )" + std::to_string(periodogram_length_) + R"(;
            overlap = 0;
      )" + periodogram_config(estimator) + R"(
          };
        }
      );
    )";
  }

  static std::string density_config(
      const std::string& estimator,
      const std::string& spatial_backend,
      const std::string& time_backend,
      const int cuda_memory_limit_mib,
      const std::string& circular_channels) {
    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 2.5e-13;
        t_min  = 2.5e-13;
        t_max  = 2.0e-12;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; },
        { name = "B"; moment = 1.5; spin = [0.0, 0.0, 1.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.00, 0.0, 0.0]),
          ("B", [0.50, 0.0, 0.0])
        );
      };

      lattice : {
        size = [2, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };

      monitors = (
        {
          module = "magnon-density";
          output_steps = 1;
      )") + circular_channels_config(circular_channels) + std::string(R"(
          keep_negative_frequencies = false;
          fftw_threads = 1;
          sk_time_series_backend = "memory";
          spatial_fft_backend = ")") + spatial_backend + R"(";
          time_fft_backend = ")" + time_backend + R"(";
          cuda_time_fft_memory_limit_mib = )" + std::to_string(cuda_memory_limit_mib) + R"(;
          compute_periodogram : {
            length = )" + std::to_string(periodogram_length_) + R"(;
            overlap = 0;
      )" + periodogram_config(estimator) + R"(
          };
        }
      );
    )";
  }

  static std::string one_basis_density_config(const std::string& circular_channels) {
    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 2.5e-13;
        t_min  = 2.5e-13;
        t_max  = 2.0e-12;
      };

      materials = (
        { name = "A"; moment = )") + std::to_string(one_basis_moment_mu_b_) + R"(; spin = [0.0, 0.0, 1.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.00, 0.0, 0.0])
        );
      };

      lattice : {
        size = [1, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };

      monitors = (
        {
          module = "magnon-density";
          output_steps = 1;
      )" + circular_channels_config(circular_channels) + R"(
          keep_negative_frequencies = false;
          fftw_threads = 1;
          sk_time_series_backend = "memory";
          spatial_fft_backend = "cpu";
          time_fft_backend = "cpu";
          compute_periodogram : {
            length = )" + std::to_string(periodogram_length_) + R"(;
            overlap = 0;
            estimator = "welch";
          };
        }
      );
    )";
  }

  static std::string circular_channels_config(const std::string& circular_channels) {
    if (circular_channels.empty()) {
      return "";
    }
    return "          circular_channels = \"" + circular_channels + "\";\n";
  }

  static std::string cartesian_probe_config(
      const std::string& spatial_backend,
      const std::string& time_backend,
      const std::string& estimator) {
    return std::string(R"(
      solver : {
        module = "llg-heun-cpu";
        t_step = 2.5e-13;
        t_min  = 2.5e-13;
        t_max  = 2.0e-12;
      };

      materials = (
        { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; },
        { name = "B"; moment = 1.5; spin = [0.0, 0.0, 1.0]; }
      );

      unitcell : {
        symops = false;
        parameter = 1.0e-9;
        basis = (
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]);
        positions = (
          ("A", [0.00, 0.0, 0.0]),
          ("B", [0.50, 0.0, 0.0])
        );
      };

      lattice : {
        size = [2, 1, 1];
        periodic = [true, true, true];
        normalise_spins = false;
      };

      monitors = (
        {
          module = "spectrum-probe";
          output_steps = 1;
          keep_negative_frequencies = false;
          fftw_threads = 1;
          sk_time_series_backend = "memory";
          spatial_fft_backend = ")") + spatial_backend + R"(";
          time_fft_backend = ")" + time_backend + R"(";
          hkl_path = (
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]
          );
          compute_periodogram : {
            length = )" + std::to_string(periodogram_length_) + R"(;
            overlap = 0;
      )" + periodogram_config(estimator) + R"(
          };
        }
      );
    )";
  }

  static constexpr int periodogram_length_ = 8;
  static constexpr double one_basis_moment_mu_b_ = 2.75;
  static constexpr double one_basis_parameter_m_ = 1.0e-9;
  static constexpr double one_basis_transverse_scale_ = 0.02;
  std::filesystem::path output_dir_;
};

TEST_F(MagnonSpectrumCudaMonitorTest, AutoBackendsWithCpuSolverProduceFiniteSpectrum) {
  MagnonSpectrumStubSolver solver;
  const auto rows = run_spectrum(solver, "cpu_auto", "welch", "auto", "auto");
  expect_finite_spectrum(rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, RejectsCudaTimeWithCpuSpatialBackend) {
  MagnonSpectrumStubSolver solver;
  initialise_lattice("welch", "cpu", "cuda");
  globals::solver = &solver;
  EXPECT_THROW((void)MagnonSpectrumMonitor(first_monitor_settings()), std::runtime_error);
  globals::solver = nullptr;
}

TEST_F(MagnonSpectrumCudaMonitorTest, RejectsExplicitCudaSpatialWithoutCudaSolver) {
  MagnonSpectrumStubSolver solver;
  initialise_lattice("welch", "cuda", "cpu");
  globals::solver = &solver;
  EXPECT_THROW((void)MagnonSpectrumMonitor(first_monitor_settings()), std::runtime_error);
  globals::solver = nullptr;
}

TEST_F(MagnonSpectrumCudaMonitorTest, MagnonDensityWritesTwoSidedAndFoldedRows) {
  MagnonSpectrumStubSolver solver;
  const auto rows = run_density(solver, "density_cpu_welch", "welch", "cpu", "cpu");
  expect_density_header_equals(
      read_density_header(),
      {"f_THz",
       "E_meV",
       "magnon_density_two_sided_meV^-1_m^-3",
       "magnon_density_positive_folded_meV^-1_m^-3"});
  expect_finite_density(rows);
  ASSERT_EQ(rows.size(), static_cast<std::size_t>(periodogram_length_));

  const double df = 1.0 / (static_cast<double>(periodogram_length_) * solver.time_step());
  expect_density_folded_columns(rows, 2, 3, df);
}

TEST_F(MagnonSpectrumCudaMonitorTest, MagnonDensityExplicitPlusMatchesDefaultRows) {
  MagnonSpectrumStubSolver default_solver;
  const auto default_rows = run_density(default_solver, "density_default_plus", "welch", "cpu", "cpu");
  const auto default_header = read_density_header();

  MagnonSpectrumStubSolver explicit_solver;
  const auto explicit_rows = run_density(explicit_solver, "density_explicit_plus", "welch", "cpu", "cpu", 0, "plus");
  const auto explicit_header = read_density_header();

  expect_density_header_equals(explicit_header, default_header);
  expect_spectra_near(explicit_rows, default_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, MagnonDensityBothCircularChannelsWritesTwoSidedAndFoldedRows) {
  MagnonSpectrumStubSolver solver;
  const auto rows = run_density(solver, "density_cpu_both_welch", "welch", "cpu", "cpu", 0, "both");
  expect_density_header_equals(
      read_density_header(),
      {"f_THz",
       "E_meV",
       "magnon_density_S+_two_sided_meV^-1_m^-3",
       "magnon_density_S-_two_sided_meV^-1_m^-3",
       "magnon_density_S+_positive_folded_meV^-1_m^-3",
       "magnon_density_S-_positive_folded_meV^-1_m^-3"});
  expect_finite_density(rows, 6);
  ASSERT_EQ(rows.size(), static_cast<std::size_t>(periodogram_length_));

  const double df = 1.0 / (static_cast<double>(periodogram_length_) * solver.time_step());
  expect_density_folded_columns(rows, 2, 4, df);
  expect_density_folded_columns(rows, 3, 5, df);

  bool found_distinct_channels = false;
  for (const auto& row : rows) {
    const double scale = std::max({1.0, std::abs(row[2]), std::abs(row[3])});
    found_distinct_channels = found_distinct_channels || std::abs(row[2] - row[3]) > 1.0e-10 * scale;
  }
  EXPECT_TRUE(found_distinct_channels);
}

TEST_F(MagnonSpectrumCudaMonitorTest, MagnonDensityUsesDimensionlessPerBasisSpinLength) {
  MagnonSpectrumStubSolver probe_solver;
  const auto probe = run_raised_probe(probe_solver, "welch", "cpu", "cpu");
  ASSERT_EQ(probe.spin_lengths.size(), 2u);
  EXPECT_NEAR(probe.spin_lengths[0], 1.0 / kElectronGFactor, 1.0e-12);
  EXPECT_NEAR(probe.spin_lengths[1], 1.5 / kElectronGFactor, 1.0e-12);

  MagnonSpectrumStubSolver density_solver;
  const auto rows = run_density(density_solver, "density_cpu_weighted", "welch", "cpu", "cpu");
  expect_finite_density(rows);

  std::vector<double> expected_by_frequency(static_cast<std::size_t>(periodogram_length_), 0.0);
  for (const auto& power : probe.powers) {
    expected_by_frequency[static_cast<std::size_t>(power.frequency)] +=
        power.power / probe.spin_lengths[static_cast<std::size_t>(power.basis)];
  }

  const double df = 1.0 / (static_cast<double>(periodogram_length_) * density_solver.time_step());
  const double volume = 2.0 * pow3(1.0e-9);
  const double prefactor = 1.0 / (volume * df * kTHz2meV);
  for (const auto& row : rows) {
    const int freq_bin = static_cast<int>(std::llround(row[0] / df));
    const int f = (freq_bin >= 0) ? freq_bin : periodogram_length_ + freq_bin;
    ASSERT_GE(f, 0);
    ASSERT_LT(f, periodogram_length_);
    const double expected = prefactor * expected_by_frequency[static_cast<std::size_t>(f)];
    EXPECT_NEAR(row[2], expected, std::max(1.0, std::abs(expected)) * 1.0e-6);
  }
}

TEST_F(MagnonSpectrumCudaMonitorTest, MagnonDensityOneBasisMatchesHandComputedDensity) {
  MagnonSpectrumStubSolver solver;
  const auto rows = run_one_basis_density(solver, "density_one_basis_hand_computed");
  expect_finite_density(rows);
  ASSERT_EQ(rows.size(), static_cast<std::size_t>(periodogram_length_));

  const double df = 1.0 / (static_cast<double>(periodogram_length_) * solver.time_step());
  const auto expected = expected_one_basis_density_two_sided(solver.time_step(), 0);
  for (const auto& row : rows) {
    const int freq_bin = static_cast<int>(std::llround(row[0] / df));
    const int f = (freq_bin >= 0) ? freq_bin : periodogram_length_ + freq_bin;
    ASSERT_GE(f, 0);
    ASSERT_LT(f, periodogram_length_);

    const double expected_two_sided = expected[static_cast<std::size_t>(f)];
    EXPECT_NEAR(row[2], expected_two_sided, std::max(1.0, std::abs(expected_two_sided)) * 1.0e-5);

    double expected_folded = 0.0;
    if (freq_bin == 0 || (periodogram_length_ % 2 == 0 && f == periodogram_length_ / 2)) {
      expected_folded = expected_two_sided;
    } else if (freq_bin > 0) {
      expected_folded = expected[static_cast<std::size_t>(f)]
          + expected[static_cast<std::size_t>(periodogram_length_ - f)];
    }
    EXPECT_NEAR(row[3], expected_folded, std::max(1.0, std::abs(expected_folded)) * 1.0e-5);
  }
}

TEST_F(MagnonSpectrumCudaMonitorTest, MagnonDensityBothChannelsOneBasisMatchesHandComputedDensity) {
  MagnonSpectrumStubSolver solver;
  const auto rows = run_one_basis_density(solver, "density_one_basis_both_hand_computed", "both");
  expect_finite_density(rows, 6);
  ASSERT_EQ(rows.size(), static_cast<std::size_t>(periodogram_length_));

  const double df = 1.0 / (static_cast<double>(periodogram_length_) * solver.time_step());
  const auto expected_plus = expected_one_basis_density_two_sided(solver.time_step(), 0);
  const auto expected_minus = expected_one_basis_density_two_sided(solver.time_step(), 1);
  for (const auto& row : rows) {
    const int freq_bin = static_cast<int>(std::llround(row[0] / df));
    const int f = (freq_bin >= 0) ? freq_bin : periodogram_length_ + freq_bin;
    ASSERT_GE(f, 0);
    ASSERT_LT(f, periodogram_length_);

    const double expected_plus_two_sided = expected_plus[static_cast<std::size_t>(f)];
    const double expected_minus_two_sided = expected_minus[static_cast<std::size_t>(f)];
    EXPECT_NEAR(row[2], expected_plus_two_sided, std::max(1.0, std::abs(expected_plus_two_sided)) * 1.0e-5);
    EXPECT_NEAR(row[3], expected_minus_two_sided, std::max(1.0, std::abs(expected_minus_two_sided)) * 1.0e-5);

    double expected_plus_folded = 0.0;
    double expected_minus_folded = 0.0;
    if (freq_bin == 0 || (periodogram_length_ % 2 == 0 && f == periodogram_length_ / 2)) {
      expected_plus_folded = expected_plus_two_sided;
      expected_minus_folded = expected_minus_two_sided;
    } else if (freq_bin > 0) {
      expected_plus_folded = expected_plus[static_cast<std::size_t>(f)]
          + expected_plus[static_cast<std::size_t>(periodogram_length_ - f)];
      expected_minus_folded = expected_minus[static_cast<std::size_t>(f)]
          + expected_minus[static_cast<std::size_t>(periodogram_length_ - f)];
    }
    EXPECT_NEAR(row[4], expected_plus_folded, std::max(1.0, std::abs(expected_plus_folded)) * 1.0e-5);
    EXPECT_NEAR(row[5], expected_minus_folded, std::max(1.0, std::abs(expected_minus_folded)) * 1.0e-5);
  }
}

#if HAS_CUDA
TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCpuTimeMatchesCpuWelchSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_spectrum(cpu_solver, "cpu_welch", "welch", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_spectrum(cuda_solver, "cuda_spatial_cpu_time_welch", "welch", "cuda", "cpu");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCpuTimeMatchesCpuCartesianSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_cartesian_probe(cpu_solver, "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_cartesian_probe(cuda_solver, "cuda", "cpu");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialAutoTimeMatchesCpuCartesianSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_cartesian_probe(cpu_solver, "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_cartesian_probe(cuda_solver, "cuda", "auto");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuCartesianSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_cartesian_probe(cpu_solver, "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_cartesian_probe(cuda_solver, "cuda", "cuda");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuCartesianMultitaperSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_cartesian_probe(cpu_solver, "cpu", "cpu", "multitaper");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_cartesian_probe(cuda_solver, "cuda", "cuda", "multitaper");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuWelchSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_spectrum(cpu_solver, "cpu_welch_cuda_time_ref", "welch", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_spectrum(cuda_solver, "cuda_spatial_cuda_time_welch", "welch", "cuda", "cuda");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuWelchMagnonDensity) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_density(cpu_solver, "density_cpu_welch_cuda_ref", "welch", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_density(cuda_solver, "density_cuda_welch", "welch", "cuda", "cuda");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuWelchBothChannelMagnonDensity) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_density(cpu_solver, "density_cpu_both_welch_cuda_ref", "welch", "cpu", "cpu", 0, "both");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_density(cuda_solver, "density_cuda_both_welch", "welch", "cuda", "cuda", 0, "both");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCpuTimeMatchesCpuMultitaperSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_spectrum(cpu_solver, "cpu_multitaper", "multitaper", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_spectrum(cuda_solver, "cuda_spatial_cpu_time_multitaper", "multitaper", "cuda", "cpu");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuMultitaperSpectrum) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_spectrum(cpu_solver, "cpu_multitaper_cuda_time_ref", "multitaper", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_spectrum(cuda_solver, "cuda_spatial_cuda_time_multitaper", "multitaper", "cuda", "cuda");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuMultitaperMagnonDensity) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows = run_density(cpu_solver, "density_cpu_multitaper_cuda_ref", "multitaper", "cpu", "cpu");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows = run_density(cuda_solver, "density_cuda_multitaper", "multitaper", "cuda", "cuda");

  expect_spectra_near(cuda_rows, cpu_rows);
}

TEST_F(MagnonSpectrumCudaMonitorTest, CudaSpatialCudaTimeMatchesCpuMultitaperBothChannelMagnonDensity) {
  if (!magnon_spectrum_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MagnonSpectrumStubSolver cpu_solver;
  const auto cpu_rows =
      run_density(cpu_solver, "density_cpu_both_multitaper_cuda_ref", "multitaper", "cpu", "cpu", 0, "both");

  MagnonSpectrumCudaStubSolver cuda_solver;
  const auto cuda_rows =
      run_density(cuda_solver, "density_cuda_both_multitaper", "multitaper", "cuda", "cuda", 0, "both");

  expect_spectra_near(cuda_rows, cpu_rows);
}
#endif

}  // namespace jams::testing

#endif  // JAMS_TEST_MONITORS_TEST_MAGNON_SPECTRUM_CUDA_H
