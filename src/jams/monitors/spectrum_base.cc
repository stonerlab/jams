//
// Created by Joseph Barker on 2019-08-01.
//

#include "jams/monitors/spectrum_base.h"

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/interface/fft.h"
#include "jams/interface/lapack_tridiagonal.h"
#include "jams/monitors/kpoint_path_builder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Channel maps
// ---------------------------------------------------------------------------
SpectrumBaseMonitor::ChannelTransform SpectrumBaseMonitor::cartesian_channel_map()
{
  return ChannelTransform{};
}

SpectrumBaseMonitor::ChannelTransform SpectrumBaseMonitor::raise_lower_channel_map()
{
  ChannelTransform m;
  m.output_channels = 3;
  m.weights = {{
      {{kInvSqrtTwo, +kImagOne * kInvSqrtTwo, 0.0}},
      {{kInvSqrtTwo, -kImagOne * kInvSqrtTwo, 0.0}},
      {{0.0, 0.0, 1.0}}
  }};
  m.use_local_frame = true;
  m.scale_to_physical_spin = true;
  return m;
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------
void SpectrumBaseMonitor::configure_storage_backend_policy_(const libconfig::Setting& settings)
{
  std::string backend_setting = "auto";
  if (settings.exists("sk_time_series_backend"))
  {
    backend_setting = jams::config_required<std::string>(settings, "sk_time_series_backend");
  }
  else
  {
    // Backwards-compatible alias.
    backend_setting = jams::config_optional<std::string>(settings, "storage", backend_setting);
  }

  const auto backend_setting_lc = lowercase(backend_setting);
  if (backend_setting_lc == "auto")
  {
    sk_time_series_backend_policy_ = SkTimeSeriesBackendPolicy::Auto;
    return;
  }
  if (backend_setting_lc == "memory" || backend_setting_lc == "in_memory")
  {
    sk_time_series_backend_policy_ = SkTimeSeriesBackendPolicy::Memory;
    return;
  }
  if (backend_setting_lc == "file" || backend_setting_lc == "file_backed")
  {
    sk_time_series_backend_policy_ = SkTimeSeriesBackendPolicy::File;
    return;
  }

  throw std::runtime_error(
      "sk_time_series_backend must be one of: auto, memory, file");
}

void SpectrumBaseMonitor::initialise_k_points_(
    const libconfig::Setting& settings,
    const KSamplingMode k_sampling_mode)
{
  KPointPathBuilder builder(*globals::lattice);
  const auto kspace_size = globals::lattice->kspace_size();

  std::cout << "  creating k-point list" << std::endl;
  if (k_sampling_mode == KSamplingMode::FullGrid)
  {
    builder.append_full_k_grid(k_points_, k_segment_offsets_, kspace_size);
    full_brillouin_zone_appended_ = true;
  }
  else
  {
    full_brillouin_zone_appended_ = builder.configure_k_list(
        k_points_, k_segment_offsets_, settings["hkl_path"], kspace_size);
  }
}

void SpectrumBaseMonitor::initialise_basis_phase_factors_()
{
  std::cout << "  generating basis phase factors" << std::endl;
  std::vector<Vec3> r_frac(num_basis_atoms());
  for (auto a = 0; a < num_basis_atoms(); ++a)
  {
    r_frac[a] = globals::lattice->basis_site_atom(a).position_frac;
  }
  generate_phase_factors_(basis_phase_factors_, r_frac, k_points_);
}

SpectrumBaseMonitor::SpectrumBaseMonitor(
    const libconfig::Setting& settings,
    const KSamplingMode k_sampling_mode)
    : Monitor(settings),
      sk_time_series_(0, jams::instance().temp_directory_path())
{
  const auto kspace_size = globals::lattice->kspace_size();
  num_basis_atoms_ = globals::lattice->num_basis_sites();

  keep_negative_frequencies_ = jams::config_optional<bool>(
      settings,
      "keep_negative_frequencies",
      keep_negative_frequencies_);
  configure_storage_backend_policy_(settings);

  if (settings.exists("compute_periodogram"))
  {
    configure_periodogram(settings["compute_periodogram"]);
  }

  initialise_k_points_(settings, k_sampling_mode);
  initialise_basis_phase_factors_();

  std::cout << "  allocating sk_grid buffer" << std::endl;

  zero(sk_grid_.resize(
      kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_basis_atoms_));

  std::cout << "  deferring sk_time_series buffer allocation" << std::endl;
}

// ---------------------------------------------------------------------------
// Lifecycle and storage selection
// ---------------------------------------------------------------------------
SpectrumBaseMonitor::~SpectrumBaseMonitor()
{
  fftw_destroy_plan(sk_time_fft_plan_);
}

void SpectrumBaseMonitor::set_channel_map(const ChannelTransform& channel_map)
{
  if (channel_map.output_channels < 1 || channel_map.output_channels > 3)
  {
    throw std::runtime_error("SpectrumBaseMonitor::set_channel_map output_channels must be in [1,3]");
  }

  channel_transform_ = channel_map;
  if (sk_time_series_storage_initialised_)
  {
    resize_channel_storage_();
  }
  periodogram_sample_index_ = 0;
  periodogram_window_count_ = 1;

  if (sk_time_fft_plan_)
  {
    fftw_destroy_plan(sk_time_fft_plan_);
    sk_time_fft_plan_ = nullptr;
  }
}

bool SpectrumBaseMonitor::needs_local_frame_mapping_() const
{
  return channel_transform_.use_local_frame;
}

bool SpectrumBaseMonitor::use_file_backed_sk_time_series_() const
{
  switch (sk_time_series_backend_policy_)
  {
    case SkTimeSeriesBackendPolicy::Auto:
      return full_brillouin_zone_appended_;
    case SkTimeSeriesBackendPolicy::Memory:
      return false;
    case SkTimeSeriesBackendPolicy::File:
      return true;
  }
  throw std::runtime_error("Invalid sk_time_series backend policy");
}

void SpectrumBaseMonitor::ensure_channel_storage_initialised_()
{
  if (!sk_time_series_storage_initialised_)
  {
    resize_channel_storage_();
  }
}

void SpectrumBaseMonitor::log_channel_storage_info_() const
{
  const double sk_time_series_size_mib =
      static_cast<double>(sk_time_series_.required_bytes()) / (1024.0 * 1024.0);
  std::cout << "    sk_time_series size (MiB) " << sk_time_series_size_mib << std::endl;

  if (sk_time_series_.using_file_backed_ring_buffer())
  {
    std::cout << "    sk_time_series backend file-backed ring buffer "
              << sk_time_series_.file_path() << std::endl;
  }
  else
  {
    std::cout << "    sk_time_series backend in-memory" << std::endl;
  }
}

void SpectrumBaseMonitor::resize_channel_storage_()
{
  const int T = periodogram_props_.length;
  const int K = static_cast<int>(k_points_.size());
  const int A = num_basis_atoms_;
  const int C = num_channels();

  if (needs_local_frame_mapping_())
  {
    stored_channel_count_ = 3;
  }
  else
  {
    stored_channel_count_ = C;
  }
  sk_time_series_.resize(
      {static_cast<std::size_t>(T),
       static_cast<std::size_t>(A),
       static_cast<std::size_t>(K),
       static_cast<std::size_t>(stored_channel_count_)},
      use_file_backed_sk_time_series_());
  sk_time_series_storage_initialised_ = true;
  log_channel_storage_info_();

  if (needs_local_frame_mapping_())
  {
    basis_mag_time_series_.resize(num_basis_atoms(), periodogram_length());
    basis_mag_time_series_.zero();
  }
  else
  {
    basis_mag_time_series_.clear();
  }

  if (periodogram_window_.size() != T)
  {
    generate_normalised_window_(periodogram_window_, T);
  }

  if (temporal_estimator_ == TemporalEstimator::Multitaper)
  {
    if (multitaper_windows_.size(0) != multitaper_count_ || multitaper_windows_.size(1) != T)
    {
      generate_normalised_dpss_tapers_(multitaper_windows_, multitaper_count_, T, multitaper_bandwidth_);
    }
  }
}

// ---------------------------------------------------------------------------
// k-path configuration
// ---------------------------------------------------------------------------
void SpectrumBaseMonitor::configure_temporal_estimator_(libconfig::Setting& settings)
{
  std::string estimator = jams::config_optional<std::string>(settings, "estimator", "welch");
  estimator = lowercase(estimator);

  if (estimator == "welch")
  {
    temporal_estimator_ = TemporalEstimator::Welch;
    return;
  }

  if (estimator == "multitaper")
  {
    temporal_estimator_ = TemporalEstimator::Multitaper;
    multitaper_count_ = jams::config_optional<int>(settings, "multitaper_tapers", multitaper_count_);

    if (settings.exists("multitaper_bandwidth_thz"))
    {
      throw std::runtime_error(
          "multitaper_bandwidth_thz has been removed; use dimensionless multitaper_bandwidth");
    }

    const bool has_bandwidth = settings.exists("multitaper_bandwidth");
    const bool has_time_bandwidth = settings.exists("multitaper_time_bandwidth");
    if (has_bandwidth && has_time_bandwidth)
    {
      throw std::runtime_error(
          "Specify only one of multitaper_bandwidth or multitaper_time_bandwidth");
    }

    if (has_time_bandwidth)
    {
      // Backwards-compatible alias for older input files.
      multitaper_bandwidth_ = jams::config_required<double>(settings, "multitaper_time_bandwidth");
    }
    else
    {
      multitaper_bandwidth_ = jams::config_optional<double>(settings, "multitaper_bandwidth", multitaper_bandwidth_);
    }

    if (multitaper_bandwidth_ <= 0.0)
    {
      throw std::runtime_error("multitaper_bandwidth must be greater than zero");
    }
    if (multitaper_bandwidth_ >= static_cast<double>(periodogram_props_.length) / 2.0)
    {
      throw std::runtime_error(
          "multitaper bandwidth is too large for the configured periodogram length");
    }

    if (multitaper_count_ <= 0)
    {
      throw std::runtime_error("multitaper_tapers must be greater than zero");
    }
    if (multitaper_count_ > periodogram_props_.length)
    {
      throw std::runtime_error("multitaper_tapers must be less than or equal to periodogram length");
    }
    const int max_reasonable_tapers = static_cast<int>(std::floor(2.0 * multitaper_bandwidth_ - 1.0));
    if (max_reasonable_tapers < 1)
    {
      throw std::runtime_error("multitaper_bandwidth is too small for multitaper_tapers");
    }
    if (multitaper_count_ > max_reasonable_tapers)
    {
      throw std::runtime_error("multitaper_tapers must satisfy multitaper_tapers <= floor(2 * multitaper_bandwidth - 1)");
    }
    return;
  }

  throw std::runtime_error("compute_periodogram.estimator must be either 'welch' or 'multitaper'");
}

void SpectrumBaseMonitor::configure_periodogram(libconfig::Setting &settings)
{
  periodogram_props_.length = jams::config_required<int>(settings, "length");
  periodogram_props_.overlap = jams::config_optional<int>(settings, "overlap", periodogram_props_.length / 2);

  if (periodogram_props_.length <= 0)
  {
    throw std::runtime_error("Periodogram length must be greater than zero");
  }

  if (periodogram_props_.overlap < 0)
  {
    throw std::runtime_error("Periodogram overlap must be greater than or equal to zero");
  }

  if (periodogram_props_.overlap >= periodogram_props_.length)
  {
    throw std::runtime_error("Periodogram overlap must be less than periodogram length");
  }

  configure_temporal_estimator_(settings);
}

// ---------------------------------------------------------------------------
// Frequency-space processing
// ---------------------------------------------------------------------------
const SpectrumBaseMonitor::CmplxMappedSlice& SpectrumBaseMonitor::compute_frequency_spectrum_at_k(
  const int kpoint_index)
{
  const int num_sites = num_basis_atoms();
  const int num_time_samples = periodogram_length();
  const int channels = num_channels();
  const bool use_local_frame = needs_local_frame_mapping_();

  if (periodogram_window_.size() != num_time_samples)
  {
    generate_normalised_window_(periodogram_window_, num_time_samples);
  }
  if (temporal_estimator_ == TemporalEstimator::Multitaper
      && (multitaper_windows_.size(0) != multitaper_count_
          || multitaper_windows_.size(1) != num_time_samples))
  {
    generate_normalised_dpss_tapers_(
        multitaper_windows_, multitaper_count_, num_time_samples, multitaper_bandwidth_);
  }

  if (!sk_time_fft_plan_
      || frequency_scratch_.size(0) != num_sites
      || frequency_scratch_.size(1) != num_time_samples
      || frequency_scratch_.size(2) != channels)
  {
    if (sk_time_fft_plan_)
    {
      fftw_destroy_plan(sk_time_fft_plan_);
      sk_time_fft_plan_ = nullptr;
    }

    frequency_scratch_.resize(num_sites, num_time_samples, channels);

    const int n[1] = {num_time_samples};
    const int howmany = channels;
    const int istride = channels;
    const int ostride = channels;
    const int idist = 1;
    const int odist = 1;

    auto* dummy = FFTW_COMPLEX_CAST(&frequency_scratch_(0, 0, 0));

    sk_time_fft_plan_ = fftw_plan_many_dft(
        1,
        n,
        howmany,
        dummy,
        nullptr,
        istride,
        idist,
        dummy,
        nullptr,
        ostride,
        odist,
        FFTW_FORWARD,
        FFTW_ESTIMATE);

    assert(sk_time_fft_plan_);
  }

  if (frequency_accum_.size(0) != num_sites
      || frequency_accum_.size(1) != num_time_samples
      || frequency_accum_.size(2) != channels)
  {
    frequency_accum_.resize(num_sites, num_time_samples, channels);
  }

  if (temporal_estimator_ == TemporalEstimator::Multitaper
      && (frequency_taper_sum_.size(0) != num_sites
          || frequency_taper_sum_.size(1) != num_time_samples
          || frequency_taper_sum_.size(2) != channels))
  {
    frequency_taper_sum_.resize(num_sites, num_time_samples, channels);
  }
  if (temporal_estimator_ == TemporalEstimator::Multitaper
      && (frequency_taper_power_sum_.size(0) != num_sites
          || frequency_taper_power_sum_.size(1) != num_time_samples
          || frequency_taper_power_sum_.size(2) != channels))
  {
    frequency_taper_power_sum_.resize(num_sites, num_time_samples, channels);
  }
  const auto rotations = use_local_frame
      ? generate_sublattice_rotations_()
      : jams::MultiArray<Mat3, 1>{};
  const auto* rotations_ptr = use_local_frame ? &rotations : nullptr;

  for (auto a = 0; a < num_sites; ++a)
  {
    for (auto t = 0; t < num_time_samples; ++t)
    {
      if (use_local_frame)
      {
        const Vec3cx spin_xyz = read_cartesian_spin_(a, t, kpoint_index);
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = map_spin_component_(a, c, spin_xyz, rotations_ptr);
        }
      }
      else
      {
        for (auto c = 0; c < channels; ++c)
        {
          const auto s = sk_time_series_(t, a, kpoint_index, c);
          frequency_scratch_(a, t, c) = jams::ComplexHi{s.real(), s.imag()};
        }
      }
    }
  }

  const double time_norm = 1.0 / static_cast<double>(num_time_samples);

  for (auto a = 0; a < num_sites; ++a)
  {
    std::vector<jams::ComplexHi> sk0(channels, jams::ComplexHi{0.0, 0.0});

    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        sk0[c] += time_norm * frequency_scratch_(a, t, c);
      }
    }

    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        frequency_accum_(a, t, c) = time_norm * (frequency_scratch_(a, t, c) - sk0[c]);
      }
    }
  }

  if (temporal_estimator_ == TemporalEstimator::Welch)
  {
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto t = 0; t < num_time_samples; ++t)
      {
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = periodogram_window_(t) * frequency_accum_(a, t, c);
        }
      }
      auto* ptr = FFTW_COMPLEX_CAST(&frequency_scratch_(a, 0, 0));
      fftw_execute_dft(sk_time_fft_plan_, ptr, ptr);
    }
    return frequency_scratch_;
  }

  // Multitaper: average tapered complex spectra and preserve mean power.
  zero(frequency_taper_sum_);
  zero(frequency_taper_power_sum_);
  const double inv_tapers = 1.0 / static_cast<double>(multitaper_count_);

  for (auto taper = 0; taper < multitaper_count_; ++taper)
  {
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto t = 0; t < num_time_samples; ++t)
      {
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = multitaper_windows_(taper, t) * frequency_accum_(a, t, c);
        }
      }

      auto* ptr = FFTW_COMPLEX_CAST(&frequency_scratch_(a, 0, 0));
      fftw_execute_dft(sk_time_fft_plan_, ptr, ptr);

      for (auto t = 0; t < num_time_samples; ++t)
      {
        for (auto c = 0; c < channels; ++c)
        {
          frequency_taper_sum_(a, t, c) += inv_tapers * frequency_scratch_(a, t, c);
          frequency_taper_power_sum_(a, t, c) += inv_tapers * std::norm(frequency_scratch_(a, t, c));
        }
      }
    }
  }

  constexpr double kPhaseEpsilon = 1e-30;
  for (auto a = 0; a < num_sites; ++a)
  {
    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        const auto mean_complex = frequency_taper_sum_(a, t, c);
        const double mean_power = std::max(0.0, frequency_taper_power_sum_(a, t, c));
        const double mean_abs = std::abs(mean_complex);
        if (mean_abs > kPhaseEpsilon)
        {
          frequency_scratch_(a, t, c) = (std::sqrt(mean_power) / mean_abs) * mean_complex;
        }
        else
        {
          frequency_scratch_(a, t, c) = jams::ComplexHi{0.0, 0.0};
        }
      }
    }
  }
  return frequency_scratch_;
}

bool SpectrumBaseMonitor::periodogram_window_complete() const
{
  return periodogram_sample_index_ >= periodogram_props_.length && periodogram_props_.length > 0;
}

void SpectrumBaseMonitor::advance_periodogram_window()
{
  const std::size_t overlap = static_cast<std::size_t>(periodogram_overlap());

  // Keep only the overlap tail of S(k,t) as the head of the next window.
  const std::size_t num_time = sk_time_series_.size(0);

  assert(overlap < num_time);
  sk_time_series_.advance_ring_window(overlap);

  if (needs_local_frame_mapping_())
  {
    // Keep overlap for per-sublattice magnetisation and clear the non-overlap region.
    const std::size_t num_sublattices = globals::lattice->num_basis_sites();
    const std::size_t num_period_samples = static_cast<std::size_t>(periodogram_length());
    if (num_period_samples > 0)
    {
      assert(overlap < num_period_samples);

      const std::size_t mag_source0 = num_period_samples - overlap;

      for (std::size_t sublattice = 0; sublattice < num_sublattices; ++sublattice)
      {
        auto* dst = &basis_mag_time_series_(sublattice, 0);
        const auto* src = &basis_mag_time_series_(sublattice, mag_source0);
        std::copy_n(src, overlap, dst);
        std::fill_n(&basis_mag_time_series_(sublattice, overlap),
                    num_period_samples - overlap,
                    Vec3{0, 0, 0});
      }
    }
  }

  // Reset write index to the overlap boundary for the next incoming sample.
  periodogram_sample_index_ = periodogram_props_.overlap;
  periodogram_window_count_++;
}


// TODO: Remove this an refactor NeutronSpectrum to stream over k
const SpectrumBaseMonitor::CmplxMappedSpectrum& SpectrumBaseMonitor::finalise_periodogram_spectrum()
{
  const int num_sites = num_basis_atoms();
  const int num_k = num_k_points();
  const int channels = num_channels();
  const int num_freq_out = keep_negative_frequencies_ ? periodogram_length() : (periodogram_length() / 2 + 1);

  if (skw_buffer_.size(0) != num_sites
      || skw_buffer_.size(1) != num_freq_out
      || skw_buffer_.size(2) != num_k
      || skw_buffer_.size(3) != channels)
  {
    skw_buffer_.resize(num_sites, num_freq_out, num_k, channels);
  }

  for (auto k = 0; k < num_k; ++k)
  {
    auto& sw_spectrum = compute_frequency_spectrum_at_k(k);
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto f = 0; f < num_freq_out; ++f)
      {
        for (auto c = 0; c < channels; ++c)
        {
          skw_buffer_(a, f, k, c) = sw_spectrum(a, f, c);
        }
      }
    }
  }

  advance_periodogram_window();
  return skw_buffer_;
}

void SpectrumBaseMonitor::append_sk_sample_for_k_list(const jams::MultiArray<Vec3cx,4> &sk_sample,
                                                    const std::vector<jams::HKLIndex> &k_list)
{
  const auto time_index = static_cast<std::size_t>(periodogram_sample_index_);
  const auto num_basis = static_cast<std::size_t>(sk_sample.size(3));
  const auto num_k = k_list.size();
  const auto stored_channels = static_cast<std::size_t>(stored_channel_count_);
  const bool use_local_frame = needs_local_frame_mapping_();
  std::vector<CmplxStored> tail_buffer(num_basis * num_k * stored_channels);

  for (auto a = 0; a < sk_sample.size(3); ++a)
  {
    for (auto k = 0; k < k_list.size(); ++k)
    {
      const auto [offset, is_conjugate] = k_list[k].index;
      const auto idx = offset;
      const auto base =
          (static_cast<std::size_t>(a) * num_k + static_cast<std::size_t>(k)) * stored_channels;

      Vec3cx spin_xyz;
      if (is_conjugate)
      {
        spin_xyz = basis_phase_factors_(a, k) * jams::conj(sk_sample(idx[0], idx[1], idx[2], a));
      }
      else
      {
        spin_xyz = basis_phase_factors_(a, k) * sk_sample(idx[0], idx[1], idx[2], a);
      }

      if (use_local_frame)
      {
        tail_buffer[base + 0] = CmplxStored{static_cast<float>(spin_xyz[0].real()), static_cast<float>(spin_xyz[0].imag())};
        tail_buffer[base + 1] = CmplxStored{static_cast<float>(spin_xyz[1].real()), static_cast<float>(spin_xyz[1].imag())};
        tail_buffer[base + 2] = CmplxStored{static_cast<float>(spin_xyz[2].real()), static_cast<float>(spin_xyz[2].imag())};
      }
      else
      {
        for (std::size_t c = 0; c < stored_channels; ++c)
        {
          const auto value = map_spin_component_(a, static_cast<int>(c), spin_xyz, nullptr);
          tail_buffer[base + c] = CmplxStored{static_cast<float>(value.real()), static_cast<float>(value.imag())};
        }
      }
    }
  }

  const std::array<std::size_t, 1> prefix{time_index};
  sk_time_series_.for_each_tail_block<3>(
      prefix,
      [&](CmplxStored* destination, const std::size_t logical_offset, const std::size_t count)
      {
        std::copy_n(tail_buffer.data() + logical_offset, count, destination);
      });
}

// ---------------------------------------------------------------------------
// Local-frame mapping and accumulation
// ---------------------------------------------------------------------------
void SpectrumBaseMonitor::store_sublattice_magnetisation_(const jams::MultiArray<double, 2>& spin_state)
{
  if (basis_mag_time_series_.empty())
  {
    basis_mag_time_series_.resize(num_basis_atoms(), periodogram_length());
    basis_mag_time_series_.zero();
  }
  const auto p = periodogram_sample_index();
  for (auto i = 0; i < globals::num_spins; ++i)
  {
    Vec3 spin = {spin_state(i, 0), spin_state(i, 1), spin_state(i, 2)};
    const auto m = globals::lattice->lattice_site_basis_index(i);
    basis_mag_time_series_(m, p) += spin;
  }
}

jams::MultiArray<Vec3, 1> SpectrumBaseMonitor::compute_mean_basis_mag_directions_()
{
  if (periodogram_window_.empty())
  {
    generate_normalised_window_(periodogram_window_, periodogram_length());
  }

  jams::MultiArray<Vec3, 1> mean_directions(num_basis_atoms());
  zero(mean_directions);

  for (auto m = 0; m < num_basis_atoms(); ++m)
  {
    for (auto n = 0; n < periodogram_length(); ++n)
    {
      mean_directions(m) += periodogram_window_(n) * basis_mag_time_series_(m, n);
    }

    mean_directions(m) = jams::normalize(mean_directions(m));
  }

  return mean_directions;
}

jams::MultiArray<Mat3, 1> SpectrumBaseMonitor::generate_sublattice_rotations_()
{
  jams::MultiArray<Mat3, 1> rotations(num_basis_atoms());
  for (auto a = 0; a < num_basis_atoms(); ++a)
  {
    rotations(a) = kIdentityMat3;
  }

  if (!channel_transform_.use_local_frame)
  {
    return rotations;
  }

  const auto mean_directions = compute_mean_basis_mag_directions_();

  for (auto m = 0; m < mean_directions.size(); ++m)
  {
    Vec3 n_hat = mean_directions(m);
    const double n_norm = jams::norm(n_hat);
    if (n_norm <= 0.0)
    {
      rotations(m) = kIdentityMat3;
      continue;
    }
    n_hat *= (1.0 / n_norm);

    const Vec3 ex{1.0, 0.0, 0.0};
    const Vec3 ey{0.0, 1.0, 0.0};
    const Vec3 ez{0.0, 0.0, 1.0};

    const double ax = std::abs(jams::dot(ex, n_hat));
    const double ay = std::abs(jams::dot(ey, n_hat));
    const double az = std::abs(jams::dot(ez, n_hat));

    Vec3 r = ex;
    double a_min = ax;
    if (ay < a_min)
    {
      r = ey;
      a_min = ay;
    }
    if (az < a_min)
    {
      r = ez;
      a_min = az;
    }

    Vec3 e1 = r - jams::dot(r, n_hat) * n_hat;
    const double e1_norm = jams::norm(e1);
    if (e1_norm <= 0.0)
    {
      rotations(m) = kIdentityMat3;
      continue;
    }
    e1 *= (1.0 / e1_norm);

    Vec3 e2 = jams::cross(n_hat, e1);

    Mat3 R = kIdentityMat3;
    R[0][0] = e1[0];    R[0][1] = e1[1];    R[0][2] = e1[2];
    R[1][0] = e2[0];    R[1][1] = e2[1];    R[1][2] = e2[2];
    R[2][0] = n_hat[0]; R[2][1] = n_hat[1]; R[2][2] = n_hat[2];

    rotations(m) = R;
  }

  return rotations;
}

jams::ComplexHi SpectrumBaseMonitor::map_spin_component_(
    const int basis_index,
    const int channel_index,
    const Vec3cx& spin_xyz,
    const jams::MultiArray<Mat3, 1>* rotations) const
{
  Vec3cx s = spin_xyz;

  if (rotations)
  {
    s = (*rotations)(basis_index) * s;
  }

  if (channel_transform_.scale_to_physical_spin)
  {
    const double mu = globals::mus(basis_index);
    const double spin_length = mu / kElectronGFactor;
    s *= spin_length;
  }

  const auto& w = channel_transform_.weights[channel_index];
  return w[0] * s[0] + w[1] * s[1] + w[2] * s[2];
}

Vec3cx SpectrumBaseMonitor::read_cartesian_spin_(const int basis_index,
                                                 const int time_index,
                                                 const int k_index) const
{
  assert(stored_channel_count_ >= 3);
  const auto sx = sk_time_series_(time_index, basis_index, k_index, 0);
  const auto sy = sk_time_series_(time_index, basis_index, k_index, 1);
  const auto sz = sk_time_series_(time_index, basis_index, k_index, 2);
  return Vec3cx{
      jams::ComplexHi{sx.real(), sx.imag()},
      jams::ComplexHi{sy.real(), sy.imag()},
      jams::ComplexHi{sz.real(), sz.imag()}
  };
}

void SpectrumBaseMonitor::generate_normalised_window_(jams::MultiArray<double, 1>& window, int num_time_samples)
{
  if (window.size() != num_time_samples)
  {
    window.resize(num_time_samples);
  }

  double w2sum = 0.0;
  for (int i = 0; i < num_time_samples; ++i)
  {
    const double w = fft_window_default(i, num_time_samples);
    window(i) = w;
    w2sum += w * w;
  }

  const double inv_rms =
      (w2sum > 0.0) ? 1.0 / std::sqrt(w2sum / static_cast<double>(num_time_samples)) : 1.0;

  for (int i = 0; i < num_time_samples; ++i)
  {
    window(i) *= inv_rms;
  }
}

void SpectrumBaseMonitor::generate_normalised_dpss_tapers_(
    jams::MultiArray<double, 2>& tapers,
    const int num_tapers,
    const int num_time_samples,
    const double time_bandwidth)
{
  if (num_tapers <= 0)
  {
    throw std::runtime_error("num_tapers must be greater than zero");
  }
  if (num_time_samples <= 0)
  {
    throw std::runtime_error("num_time_samples must be greater than zero");
  }
  if (num_tapers > num_time_samples)
  {
    throw std::runtime_error("num_tapers must be less than or equal to num_time_samples");
  }
  if (time_bandwidth <= 0.0)
  {
    throw std::runtime_error("time_bandwidth must be greater than zero");
  }
  if (time_bandwidth >= static_cast<double>(num_time_samples) / 2.0)
  {
    throw std::runtime_error("time_bandwidth must be less than num_time_samples / 2");
  }

  if (tapers.size(0) != num_tapers || tapers.size(1) != num_time_samples)
  {
    tapers.resize(num_tapers, num_time_samples);
  }

  const int N = num_time_samples;
  const int K = num_tapers;
  const double W = time_bandwidth / static_cast<double>(N);
  const double cos_2piW = std::cos(kTwoPi * W);

  std::vector<double> diag(static_cast<std::size_t>(N), 0.0);
  std::vector<double> off(static_cast<std::size_t>(std::max(1, N - 1)), 0.0);
  for (int n = 0; n < N; ++n)
  {
    const double x = 0.5 * static_cast<double>(N - 1 - 2 * n);
    diag[n] = x * x * cos_2piW;
  }
  for (int n = 0; n < N - 1; ++n)
  {
    const auto m = static_cast<double>(n + 1);
    off[n] = 0.5 * m * static_cast<double>(N - (n + 1));
  }

  std::vector<double> eigenvectors;
  jams::solve_symmetric_tridiagonal_top_eigenvectors(diag, off, eigenvectors, N, K);

  const double rms_scale = std::sqrt(static_cast<double>(N)); // unit RMS as used for Welch window
  for (int out_k = 0; out_k < K; ++out_k)
  {
    // Selected eigenpairs are returned in ascending order; pick descending.
    const int src_k = K - 1 - out_k;

    // Deterministic sign convention for reproducibility.
    double sign = 1.0;
    for (int n = 0; n < N; ++n)
    {
      const double v = eigenvectors[static_cast<std::size_t>(n) + static_cast<std::size_t>(src_k) * static_cast<std::size_t>(N)];
      if (std::abs(v) > 1e-14)
      {
        sign = (v >= 0.0) ? 1.0 : -1.0;
        break;
      }
    }

    for (int n = 0; n < N; ++n)
    {
      const double v = eigenvectors[static_cast<std::size_t>(n) + static_cast<std::size_t>(src_k) * static_cast<std::size_t>(N)];
      tapers(out_k, n) = sign * rms_scale * v;
    }
  }
}

void SpectrumBaseMonitor::generate_phase_factors_(
    jams::MultiArray<jams::ComplexHi, 2>& phase_factors,
    const std::vector<Vec3>& r_frac,
    const std::vector<jams::HKLIndex>& kpoints)
{
  if (phase_factors.size(0) != r_frac.size() || phase_factors.size(1) != kpoints.size())
  {
    phase_factors.resize(r_frac.size(), kpoints.size());
  }

  for (auto a = 0; a < r_frac.size(); ++a)
  {
    const auto& r = r_frac[a];
    for (auto k = 0; k < kpoints.size(); ++k)
    {
      const auto& q = kpoints[k].hkl;
      phase_factors(a, k) = exp(-kImagTwoPi * jams::dot(q, r));
    }
  }
}

void SpectrumBaseMonitor::store_sk_snapshot(const jams::MultiArray<double, 2> &data)
{
  ensure_channel_storage_initialised_();

  fft_supercell_vector_field_to_kspace(
      data,
      sk_grid_,
      globals::lattice->size(),
      globals::lattice->kspace_size(),
      globals::lattice->num_basis_sites());

  if (needs_local_frame_mapping_())
  {
    store_sublattice_magnetisation_(data);
  }
  append_sk_sample_for_k_list(sk_grid_, k_points_);
  periodogram_sample_index_++;
}

void SpectrumBaseMonitor::print_info() const
{
  std::cout << "\n";
  std::cout << "  number of samples " << periodogram_length() << "\n";
  std::cout << "  sampling time (s) " << sample_time_interval() << "\n";
  std::cout << "  acquisition time (s) " << sample_time_interval() * periodogram_length() << "\n";
  std::cout << "  frequency resolution (THz) " << frequency_resolution_thz() << "\n";
  std::cout << "  maximum frequency (THz) " << max_frequency_thz() << "\n";
  std::cout << "  channels " << num_channels() << "\n";
  if (temporal_estimator_ == TemporalEstimator::Welch)
  {
    std::cout << "  temporal estimator welch\n";
  }
  else
  {
    std::cout << "  temporal estimator multitaper\n";
    std::cout << "  multitaper tapers " << multitaper_count_ << "\n";
    std::cout << "  multitaper bandwidth (NW) " << multitaper_bandwidth_ << "\n";
  }
  std::cout << "\n";
}
