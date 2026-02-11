//
// Created by Joseph Barker on 2019-08-01.
//

#include "jams/interface/fft.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/monitors/spectrum_base.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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

SpectrumBaseMonitor::SpectrumBaseMonitor(
    const libconfig::Setting &settings,
    KSamplingMode k_sampling_mode) : Monitor(settings)
{
  keep_negative_frequencies_ = jams::config_optional<bool>(settings, "keep_negative_frequencies", keep_negative_frequencies_);

  if (k_sampling_mode == KSamplingMode::FullGrid)
  {
    append_full_k_grid(globals::lattice->kspace_size());
  }
  else
  {
    configure_k_list(settings["hkl_path"]);
  }

  if (settings.exists("compute_periodogram"))
  {
    configure_periodogram(settings["compute_periodogram"]);
  }

  const auto kspace_size = globals::lattice->kspace_size();
  num_basis_atoms_ = globals::lattice->num_basis_sites();

  zero(sk_grid_.resize(
      kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_basis_atoms_));

  resize_channel_storage_();

  std::vector<Vec3> r_frac;
  for (auto a = 0; a < num_basis_atoms(); ++a)
  {
    r_frac.push_back(globals::lattice->basis_site_atom(a).position_frac);
  }
  basis_phase_factors_ = generate_phase_factors_(r_frac, k_points_);
}

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
  resize_channel_storage_();

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
  zero(sk_time_series_.resize(A, T, K, stored_channel_count_));

  const int F = keep_negative_frequencies_ ? T : (T / 2 + 1);
  zero(skw_buffer_.resize(A, F, K, C));

  if (periodogram_window_.size() != T)
  {
    periodogram_window_ = generate_normalised_window_(T);
  }
}

void SpectrumBaseMonitor::append_full_k_grid(Vec3i kspace_size)
{
  std::vector<jams::HKLIndex> hkl_path;

  for (auto l = 0; l < kspace_size[0]; ++l)
  {
    for (auto m = 0; m < kspace_size[1]; ++m)
    {
      for (auto n = 0; n < kspace_size[2]; ++n)
      {
        Vec3i coordinate = {l, m, n};
        Vec3 hkl = hadamard_product(coordinate, 1.0 / to_double(kspace_size));
        Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
        hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(coordinate, kspace_size)});
      }
    }
  }

  k_points_.insert(end(k_points_), begin(hkl_path), end(hkl_path));

  if (k_segment_offsets_.empty())
  {
    k_segment_offsets_.push_back(0);
  }
  k_segment_offsets_.push_back(k_segment_offsets_.back() + static_cast<int>(hkl_path.size()));
}

void SpectrumBaseMonitor::append_k_path_segment(libconfig::Setting& settings)
{
  if (!settings.isList())
  {
    throw std::runtime_error("SpectrumBaseMonitor::configure_continuous_kpath failed because settings is not a List");
  }

  std::vector<Vec3> hkl_path_nodes(settings.getLength());
  for (auto i = 0; i < settings.getLength(); ++i)
  {
    if (!settings[i].isArray())
    {
      throw std::runtime_error("SpectrumBaseMonitor::configure_continuous_kpath failed hkl node is not an Array");
    }

    hkl_path_nodes[i] = Vec3{settings[i][0], settings[i][1], settings[i][2]};
  }

  for (auto i = 1; i < hkl_path_nodes.size(); ++i)
  {
    if (hkl_path_nodes[i] == hkl_path_nodes[i - 1])
    {
      throw std::runtime_error("Two consecutive hkl_nodes cannot be the same");
    }
  }

  auto new_path = make_hkl_path(hkl_path_nodes, globals::lattice->kspace_size());
  k_points_.insert(end(k_points_), begin(new_path), end(new_path));

  if (k_segment_offsets_.empty())
  {
    k_segment_offsets_.push_back(0);
  }
  k_segment_offsets_.push_back(k_segment_offsets_.back() + static_cast<int>(new_path.size()));
}

void SpectrumBaseMonitor::configure_k_list(libconfig::Setting& settings)
{
  if (settings.isString() && std::string(settings.c_str()) == "full")
  {
    append_full_k_grid(globals::lattice->kspace_size());
    return;
  }

  if (settings[0].isArray())
  {
    append_k_path_segment(settings);
    return;
  }

  if (settings[0].isList())
  {
    for (auto n = 0; n < settings.getLength(); ++n)
    {
      if (settings[n].isArray())
      {
        append_k_path_segment(settings[n]);
        continue;
      }
      if (settings[n].isString() && std::string(settings[n].c_str()) == "full")
      {
        append_full_k_grid(globals::lattice->kspace_size());
        continue;
      }
      throw std::runtime_error("SpectrumBaseMonitor::configure_k_list failed because a nodes is not an Array or String");
    }
    return;
  }

  throw std::runtime_error("SpectrumBaseMonitor::configure_k_list failed because settings is not an Array, List or String");
}

void SpectrumBaseMonitor::configure_periodogram(libconfig::Setting &settings)
{
  periodogram_props_.length = jams::config_required<int>(settings, "length");
  periodogram_props_.overlap = jams::config_optional<int>(settings, "overlap", periodogram_props_.length / 2);

  if (periodogram_props_.length <= 0)
  {
    throw std::runtime_error("Periodogram length must be greater than zero");
  }

  if (periodogram_props_.overlap <= 0)
  {
    throw std::runtime_error("Periodogram overlap must be greater than zero");
  }

  if (periodogram_props_.overlap >= periodogram_props_.length)
  {
    throw std::runtime_error("Periodogram overlap must be less than periodogram length");
  }
}

std::vector<jams::HKLIndex> SpectrumBaseMonitor::make_hkl_path(const std::vector<Vec3> &hkl_nodes,
                                                                           const Vec3i &kspace_size)
{
  std::vector<jams::HKLIndex> hkl_path;

  for (auto n = 0; n < static_cast<int>(hkl_nodes.size()) - 1; ++n)
  {
    Vec3i start = to_int(hadamard_product(hkl_nodes[n], kspace_size));
    Vec3i end = to_int(hadamard_product(hkl_nodes[n + 1], kspace_size));
    Vec3i displacement = absolute(end - start);

    Vec3i step = {
        (end[0] > start[0]) ? 1 : ((end[0] < start[0]) ? -1 : 0),
        (end[1] > start[1]) ? 1 : ((end[1] < start[1]) ? -1 : 0),
        (end[2] > start[2]) ? 1 : ((end[2] < start[2]) ? -1 : 0)};

    if (displacement[0] >= displacement[1] && displacement[0] >= displacement[2])
    {
      int p1 = 2 * displacement[1] - displacement[0];
      int p2 = 2 * displacement[2] - displacement[0];
      while (start[0] != end[0])
      {
        Vec3 hkl = hadamard_product(start, 1.0 / to_double(kspace_size));
        Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
        hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[0] += step[0];
        if (p1 >= 0)
        {
          start[1] += step[1];
          p1 -= 2 * displacement[0];
        }
        if (p2 >= 0)
        {
          start[2] += step[2];
          p2 -= 2 * displacement[0];
        }
        p1 += 2 * displacement[1];
        p2 += 2 * displacement[2];
      }
    }
    else if (displacement[1] >= displacement[0] && displacement[1] >= displacement[2])
    {
      int p1 = 2 * displacement[0] - displacement[1];
      int p2 = 2 * displacement[2] - displacement[1];
      while (start[1] != end[1])
      {
        Vec3 hkl = hadamard_product(start, 1.0 / to_double(kspace_size));
        Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
        hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[1] += step[1];
        if (p1 >= 0)
        {
          start[0] += step[0];
          p1 -= 2 * displacement[1];
        }
        if (p2 >= 0)
        {
          start[2] += step[2];
          p2 -= 2 * displacement[1];
        }
        p1 += 2 * displacement[0];
        p2 += 2 * displacement[2];
      }
    }
    else
    {
      int p1 = 2 * displacement[0] - displacement[2];
      int p2 = 2 * displacement[1] - displacement[2];
      while (start[2] != end[2])
      {
        Vec3 hkl = hadamard_product(start, 1.0 / to_double(kspace_size));
        Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
        hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[2] += step[2];
        if (p1 >= 0)
        {
          start[1] += step[1];
          p1 -= 2 * displacement[2];
        }
        if (p2 >= 0)
        {
          start[0] += step[0];
          p2 -= 2 * displacement[2];
        }
        p1 += 2 * displacement[1];
        p2 += 2 * displacement[0];
      }
    }

    Vec3 hkl = hadamard_product(end, 1.0 / to_double(kspace_size));
    Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
    hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(end, kspace_size)});
  }

  hkl_path.erase(unique(hkl_path.begin(), hkl_path.end()), hkl_path.end());
  return hkl_path;
}

const SpectrumBaseMonitor::CmplxMappedSlice& SpectrumBaseMonitor::compute_frequency_spectrum_at_k(
  const int kpoint_index)
{
  const int num_sites = num_basis_atoms();
  const int num_time_samples = periodogram_length();
  const int channels = num_channels();

  if (periodogram_window_.size() != num_time_samples)
  {
    periodogram_window_ = generate_normalised_window_(num_time_samples);
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

  const auto rotations = needs_local_frame_mapping_()
      ? generate_sublattice_rotations_()
      : jams::MultiArray<Mat3, 1>{};
  const auto* rotations_ptr = needs_local_frame_mapping_() ? &rotations : nullptr;

  for (auto a = 0; a < num_sites; ++a)
  {
    for (auto t = 0; t < num_time_samples; ++t)
    {
      if (needs_local_frame_mapping_())
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
          const auto s = sk_time_series_(a, t, kpoint_index, c);
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
        frequency_scratch_(a, t, c) = time_norm * periodogram_window_(t) * (frequency_scratch_(a, t, c) - sk0[c]);
      }
    }
  }

  for (auto a = 0; a < num_sites; ++a)
  {
    auto* ptr = FFTW_COMPLEX_CAST(&frequency_scratch_(a, 0, 0));
    fftw_execute_dft(sk_time_fft_plan_, ptr, ptr);
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
  const std::size_t num_basis = sk_time_series_.size(0);
  const std::size_t num_time = sk_time_series_.size(1);
  const std::size_t num_k = sk_time_series_.size(2);
  const std::size_t num_channels = sk_time_series_.size(3);

  assert(overlap < num_time);

  const std::size_t source_time0 = num_time - overlap;
  const std::size_t contiguous_row_size = num_k * num_channels;
  for (std::size_t basis = 0; basis < num_basis; ++basis)
  {
    for (std::size_t t = 0; t < overlap; ++t)
    {
      auto* dst = &sk_time_series_(basis, t, 0, 0);
      const auto* src = &sk_time_series_(basis, source_time0 + t, 0, 0);
      std::copy_n(src, contiguous_row_size, dst);
    }
  }

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
  for (auto a = 0; a < sk_sample.size(3); ++a)
  {
    for (auto k = 0; k < k_list.size(); ++k)
    {
      const auto [offset, conj] = k_list[k].index;
      const auto i = periodogram_sample_index_;
      const auto idx = offset;

      Vec3cx spin_xyz;
      if (conj)
      {
        spin_xyz = basis_phase_factors_(a, k) * conj(sk_sample(idx[0], idx[1], idx[2], a));
      }
      else
      {
        spin_xyz = basis_phase_factors_(a, k) * sk_sample(idx[0], idx[1], idx[2], a);
      }

      if (needs_local_frame_mapping_())
      {
        sk_time_series_(a, i, k, 0) = CmplxStored{static_cast<float>(spin_xyz[0].real()), static_cast<float>(spin_xyz[0].imag())};
        sk_time_series_(a, i, k, 1) = CmplxStored{static_cast<float>(spin_xyz[1].real()), static_cast<float>(spin_xyz[1].imag())};
        sk_time_series_(a, i, k, 2) = CmplxStored{static_cast<float>(spin_xyz[2].real()), static_cast<float>(spin_xyz[2].imag())};
      }
      else
      {
        for (auto c = 0; c < num_channels(); ++c)
        {
          const auto value = map_spin_component_(a, c, spin_xyz, nullptr);
          sk_time_series_(a, i, k, c) = CmplxStored{static_cast<float>(value.real()), static_cast<float>(value.imag())};
        }
      }
    }
  }
}

void SpectrumBaseMonitor::store_sublattice_magnetisation_(const jams::MultiArray<double, 2>& spin_state)
{
  if (basis_mag_time_series_.empty())
  {
    basis_mag_time_series_.resize(num_basis_atoms(), periodogram_length());
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
    periodogram_window_ = generate_normalised_window_(periodogram_length());
  }

  jams::MultiArray<Vec3, 1> mean_directions(num_basis_atoms());
  zero(mean_directions);

  for (auto m = 0; m < num_basis_atoms(); ++m)
  {
    for (auto n = 0; n < periodogram_length(); ++n)
    {
      mean_directions(m) += periodogram_window_(n) * basis_mag_time_series_(m, n);
    }

    mean_directions(m) = normalize(mean_directions(m));
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
    const double n_norm = norm(n_hat);
    if (n_norm <= 0.0)
    {
      rotations(m) = kIdentityMat3;
      continue;
    }
    n_hat *= (1.0 / n_norm);

    const Vec3 ex{1.0, 0.0, 0.0};
    const Vec3 ey{0.0, 1.0, 0.0};
    const Vec3 ez{0.0, 0.0, 1.0};

    const double ax = std::abs(dot(ex, n_hat));
    const double ay = std::abs(dot(ey, n_hat));
    const double az = std::abs(dot(ez, n_hat));

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

    Vec3 e1 = r - dot(r, n_hat) * n_hat;
    const double e1_norm = norm(e1);
    if (e1_norm <= 0.0)
    {
      rotations(m) = kIdentityMat3;
      continue;
    }
    e1 *= (1.0 / e1_norm);

    Vec3 e2 = cross(n_hat, e1);

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
  const auto sx = sk_time_series_(basis_index, time_index, k_index, 0);
  const auto sy = sk_time_series_(basis_index, time_index, k_index, 1);
  const auto sz = sk_time_series_(basis_index, time_index, k_index, 2);
  return Vec3cx{
      jams::ComplexHi{sx.real(), sx.imag()},
      jams::ComplexHi{sy.real(), sy.imag()},
      jams::ComplexHi{sz.real(), sz.imag()}
  };
}

jams::MultiArray<double, 1> SpectrumBaseMonitor::generate_normalised_window_(int num_time_samples)
{
  jams::MultiArray<double, 1> window(num_time_samples);

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

  return window;
}

jams::MultiArray<jams::ComplexHi, 2> SpectrumBaseMonitor::generate_phase_factors_(
    const std::vector<Vec3>& r_frac,
    const std::vector<jams::HKLIndex>& kpoints)
{
  jams::MultiArray<jams::ComplexHi, 2> phase_factors(r_frac.size(), kpoints.size());

  for (auto a = 0; a < r_frac.size(); ++a)
  {
    const auto r = r_frac[a];
    for (auto k = 0; k < kpoints.size(); ++k)
    {
      const auto q = kpoints[k].hkl;
      phase_factors(a, k) = exp(-kImagTwoPi * dot(q, r));
    }
  }
  return phase_factors;
}

void SpectrumBaseMonitor::store_sk_snapshot(const jams::MultiArray<double, 2> &data)
{
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
  std::cout << "\n";
}
