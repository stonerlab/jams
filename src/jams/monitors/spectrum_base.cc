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

SpectrumBaseMonitor::ChannelMap SpectrumBaseMonitor::cartesian_channel_map()
{
  return ChannelMap{};
}

SpectrumBaseMonitor::ChannelMap SpectrumBaseMonitor::raise_lower_channel_map()
{
  ChannelMap m;
  m.output_channels = 3;
  m.coeffs = {{
      {{kInvSqrtTwo, +kImagOne * kInvSqrtTwo, 0.0}},
      {{kInvSqrtTwo, -kImagOne * kInvSqrtTwo, 0.0}},
      {{0.0, 0.0, 1.0}}
  }};
  m.rotate_to_sublattice_frame = true;
  m.scale_by_spin_length = true;
  return m;
}

SpectrumBaseMonitor::SpectrumBaseMonitor(const libconfig::Setting &settings) : Monitor(settings)
{
  keep_negative_frequencies_ = jams::config_optional<bool>(settings, "keep_negative_frequencies", keep_negative_frequencies_);

  configure_kspace_paths(settings["hkl_path"]);

  if (settings.exists("compute_periodogram"))
  {
    configure_periodogram(settings["compute_periodogram"]);
  }

  const auto kspace_size = globals::lattice->kspace_size();
  num_motif_atoms_ = globals::lattice->num_basis_sites();

  zero(kspace_data_.resize(
      kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_motif_atoms_));

  resize_channel_storage_();

  std::vector<Vec3> r_frac;
  for (auto a = 0; a < num_motif_atoms(); ++a)
  {
    r_frac.push_back(globals::lattice->basis_site_atom(a).position_frac);
  }
  sk_phase_factors_ = generate_phase_factors_(r_frac, kspace_paths_);
}

SpectrumBaseMonitor::~SpectrumBaseMonitor()
{
  fftw_destroy_plan(sw_spectrum_fft_plan_);
}

void SpectrumBaseMonitor::set_channel_map(const ChannelMap& channel_map)
{
  if (channel_map.output_channels < 1 || channel_map.output_channels > 3)
  {
    throw std::runtime_error("SpectrumBaseMonitor::set_channel_map output_channels must be in [1,3]");
  }

  channel_map_ = channel_map;
  resize_channel_storage_();

  if (sw_spectrum_fft_plan_)
  {
    fftw_destroy_plan(sw_spectrum_fft_plan_);
    sw_spectrum_fft_plan_ = nullptr;
  }
}

bool SpectrumBaseMonitor::requires_dynamic_channel_mapping_() const
{
  return channel_map_.rotate_to_sublattice_frame;
}

void SpectrumBaseMonitor::resize_channel_storage_()
{
  const int T = periodogram_props_.length;
  const int K = static_cast<int>(kspace_paths_.size());
  const int A = num_motif_atoms_;
  const int C = channel_map_.output_channels;

  if (requires_dynamic_channel_mapping_())
  {
    zero(sk_cartesian_timeseries_.resize(A, T, K));
    sk_timeseries_ = CmplxMappedField{};
  }
  else
  {
    zero(sk_timeseries_.resize(A, T, K, C));
    sk_cartesian_timeseries_ = jams::MultiArray<Vec3cx, 3>{};
  }

  const int F = keep_negative_frequencies_ ? T : (T / 2 + 1);
  zero(skw_spectrum_.resize(A, F, K, C));

  if (sw_window_.size() != T)
  {
    sw_window_ = generate_normalised_window_(T);
  }
}

void SpectrumBaseMonitor::insert_full_kspace(Vec3i kspace_size)
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

  kspace_paths_.insert(end(kspace_paths_), begin(hkl_path), end(hkl_path));

  if (kspace_continuous_path_ranges_.empty())
  {
    kspace_continuous_path_ranges_.push_back(0);
  }
  kspace_continuous_path_ranges_.push_back(kspace_continuous_path_ranges_.back() + static_cast<int>(hkl_path.size()));
}

void SpectrumBaseMonitor::insert_continuous_kpath(libconfig::Setting& settings)
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

  auto new_path = generate_hkl_kspace_path(hkl_path_nodes, globals::lattice->kspace_size());
  kspace_paths_.insert(end(kspace_paths_), begin(new_path), end(new_path));

  if (kspace_continuous_path_ranges_.empty())
  {
    kspace_continuous_path_ranges_.push_back(0);
  }
  kspace_continuous_path_ranges_.push_back(kspace_continuous_path_ranges_.back() + static_cast<int>(new_path.size()));
}

void SpectrumBaseMonitor::configure_kspace_paths(libconfig::Setting& settings)
{
  if (settings.isString() && std::string(settings.c_str()) == "full")
  {
    insert_full_kspace(globals::lattice->kspace_size());
    return;
  }

  if (settings[0].isArray())
  {
    insert_continuous_kpath(settings);
    return;
  }

  if (settings[0].isList())
  {
    for (auto n = 0; n < settings.getLength(); ++n)
    {
      if (settings[n].isArray())
      {
        insert_continuous_kpath(settings[n]);
        continue;
      }
      if (settings[n].isString() && std::string(settings[n].c_str()) == "full")
      {
        insert_full_kspace(globals::lattice->kspace_size());
        continue;
      }
      throw std::runtime_error("SpectrumBaseMonitor::configure_kspace_paths failed because a nodes is not an Array or String");
    }
    return;
  }

  throw std::runtime_error("SpectrumBaseMonitor::configure_kspace_paths failed because settings is not an Array, List or String");
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

std::vector<jams::HKLIndex> SpectrumBaseMonitor::generate_hkl_kspace_path(const std::vector<Vec3> &hkl_nodes,
                                                                           const Vec3i &kspace_size)
{
  std::vector<jams::HKLIndex> hkl_path;

  for (auto n = 0; n < static_cast<int>(hkl_nodes.size()) - 1; ++n)
  {
    Vec3i start = to_int(hadamard_product(hkl_nodes[n], kspace_size));
    Vec3i end = to_int(hadamard_product(hkl_nodes[n + 1], kspace_size));
    Vec3i displacement = absolute(end - start);

    Vec3i step = {
        (end[0] > start[0]) ? 1 : -1,
        (end[1] > start[1]) ? 1 : -1,
        (end[2] > start[2]) ? 1 : -1};

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
    hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});
  }

  hkl_path.erase(unique(hkl_path.begin(), hkl_path.end()), hkl_path.end());
  return hkl_path;
}

SpectrumBaseMonitor::CmplxMappedSlice& SpectrumBaseMonitor::fft_sk_timeseries_to_skw(const int kpoint_index)
{
  const int num_sites = num_motif_atoms();
  const int num_time_samples = num_periodogram_samples();
  const int channels = num_channels();

  if (sw_window_.size() != num_time_samples)
  {
    sw_window_ = generate_normalised_window_(num_time_samples);
  }

  if (!sw_spectrum_fft_plan_
      || sw_spectrum_buffer_.size(0) != num_sites
      || sw_spectrum_buffer_.size(1) != num_time_samples
      || sw_spectrum_buffer_.size(2) != channels)
  {
    if (sw_spectrum_fft_plan_)
    {
      fftw_destroy_plan(sw_spectrum_fft_plan_);
      sw_spectrum_fft_plan_ = nullptr;
    }

    sw_spectrum_buffer_.resize(num_sites, num_time_samples, channels);

    const int n[1] = {num_time_samples};
    const int howmany = channels;
    const int istride = channels;
    const int ostride = channels;
    const int idist = 1;
    const int odist = 1;

    auto* dummy = FFTW_COMPLEX_CAST(&sw_spectrum_buffer_(0, 0, 0));

    sw_spectrum_fft_plan_ = fftw_plan_many_dft(
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

    assert(sw_spectrum_fft_plan_);
  }

  const auto rotations = requires_dynamic_channel_mapping_()
      ? generate_sublattice_rotations_()
      : jams::MultiArray<Mat3, 1>{};
  const auto* rotations_ptr = requires_dynamic_channel_mapping_() ? &rotations : nullptr;

  for (auto a = 0; a < num_sites; ++a)
  {
    for (auto t = 0; t < num_time_samples; ++t)
    {
      if (requires_dynamic_channel_mapping_())
      {
        const Vec3cx spin_xyz = sk_cartesian_timeseries_(a, t, kpoint_index);
        for (auto c = 0; c < channels; ++c)
        {
          sw_spectrum_buffer_(a, t, c) = map_spin_component_(a, c, spin_xyz, rotations_ptr);
        }
      }
      else
      {
        for (auto c = 0; c < channels; ++c)
        {
          sw_spectrum_buffer_(a, t, c) = sk_timeseries_(a, t, kpoint_index, c);
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
        sk0[c] += time_norm * sw_spectrum_buffer_(a, t, c);
      }
    }

    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        sw_spectrum_buffer_(a, t, c) = time_norm * sw_window_(t) * (sw_spectrum_buffer_(a, t, c) - sk0[c]);
      }
    }
  }

  for (auto a = 0; a < num_sites; ++a)
  {
    auto* ptr = FFTW_COMPLEX_CAST(&sw_spectrum_buffer_(a, 0, 0));
    fftw_execute_dft(sw_spectrum_fft_plan_, ptr, ptr);
  }

  return sw_spectrum_buffer_;
}

bool SpectrumBaseMonitor::do_periodogram_update() const
{
  return periodogram_index_ >= periodogram_props_.length && periodogram_props_.length > 0;
}

void SpectrumBaseMonitor::shift_periodogram()
{
  shift_sk_timeseries_();
  shift_sublattice_magnetisation_timeseries_();
  periodogram_index_ = periodogram_props_.overlap;
  total_periods_++;
}

SpectrumBaseMonitor::CmplxMappedSpectrum SpectrumBaseMonitor::compute_periodogram_spectrum()
{
  const int num_sites = num_motif_atoms();
  const int num_k = num_kpoints();
  const int channels = num_channels();
  const int num_freq_out = keep_negative_frequencies_ ? num_periodogram_samples() : (num_periodogram_samples() / 2 + 1);

  if (skw_spectrum_.size(0) != num_sites
      || skw_spectrum_.size(1) != num_freq_out
      || skw_spectrum_.size(2) != num_k
      || skw_spectrum_.size(3) != channels)
  {
    skw_spectrum_.resize(num_sites, num_freq_out, num_k, channels);
  }

  for (auto k = 0; k < num_k; ++k)
  {
    auto& sw_spectrum = fft_sk_timeseries_to_skw(k);
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto f = 0; f < num_freq_out; ++f)
      {
        for (auto c = 0; c < channels; ++c)
        {
          skw_spectrum_(a, f, k, c) = sw_spectrum(a, f, c);
        }
      }
    }
  }

  shift_sk_timeseries_();
  periodogram_index_ = periodogram_props_.overlap;
  total_periods_++;
  return skw_spectrum_;
}

SpectrumBaseMonitor::CmplxMappedSpectrum SpectrumBaseMonitor::compute_periodogram_rotated_spectrum()
{
  return compute_periodogram_spectrum();
}

void SpectrumBaseMonitor::shift_sk_timeseries_()
{
  const std::size_t ov = static_cast<std::size_t>(periodogram_overlap());

  if (requires_dynamic_channel_mapping_())
  {
    const std::size_t A = sk_cartesian_timeseries_.size(0);
    const std::size_t T = sk_cartesian_timeseries_.size(1);
    const std::size_t K = sk_cartesian_timeseries_.size(2);
    assert(ov < T);

    const std::size_t src0 = T - ov;
    for (std::size_t a = 0; a < A; ++a)
    {
      for (std::size_t i = 0; i < ov; ++i)
      {
        auto* dst = &sk_cartesian_timeseries_(a, i, 0);
        const auto* src = &sk_cartesian_timeseries_(a, src0 + i, 0);
        std::copy_n(src, K, dst);
      }
    }
    return;
  }

  const std::size_t A = sk_timeseries_.size(0);
  const std::size_t T = sk_timeseries_.size(1);
  const std::size_t K = sk_timeseries_.size(2);
  const std::size_t C = sk_timeseries_.size(3);

  assert(ov < T);

  const std::size_t src0 = T - ov;
  const std::size_t row_size = K * C;
  for (std::size_t a = 0; a < A; ++a)
  {
    for (std::size_t i = 0; i < ov; ++i)
    {
      auto* dst = &sk_timeseries_(a, i, 0, 0);
      const auto* src = &sk_timeseries_(a, src0 + i, 0, 0);
      std::copy_n(src, row_size, dst);
    }
  }
}

void SpectrumBaseMonitor::store_kspace_data_on_path(const jams::MultiArray<Vec3cx,4> &kspace_data,
                                                    const std::vector<jams::HKLIndex> &kspace_path)
{
  for (auto a = 0; a < kspace_data.size(3); ++a)
  {
    for (auto k = 0; k < kspace_path.size(); ++k)
    {
      const auto kindex = kspace_path[k].index;
      const auto i = periodogram_index_;
      const auto idx = kindex.offset;

      Vec3cx spin_xyz;
      if (kindex.conj)
      {
        spin_xyz = sk_phase_factors_(a, k) * conj(kspace_data(idx[0], idx[1], idx[2], a));
      }
      else
      {
        spin_xyz = sk_phase_factors_(a, k) * kspace_data(idx[0], idx[1], idx[2], a);
      }

      if (requires_dynamic_channel_mapping_())
      {
        sk_cartesian_timeseries_(a, i, k) = spin_xyz;
      }
      else
      {
        for (auto c = 0; c < num_channels(); ++c)
        {
          sk_timeseries_(a, i, k, c) = map_spin_component_(a, c, spin_xyz, nullptr);
        }
      }
    }
  }
}

void SpectrumBaseMonitor::store_sublattice_magnetisation_(const jams::MultiArray<double, 2>& spin_state)
{
  if (sublattice_magnetisation_.empty())
  {
    sublattice_magnetisation_.resize(num_motif_atoms(), num_periodogram_samples());
  }
  const auto p = periodogram_index();
  for (auto i = 0; i < globals::num_spins; ++i)
  {
    Vec3 spin = {spin_state(i, 0), spin_state(i, 1), spin_state(i, 2)};
    const auto m = globals::lattice->lattice_site_basis_index(i);
    sublattice_magnetisation_(m, p) += spin;
  }
}

void SpectrumBaseMonitor::shift_sublattice_magnetisation_timeseries_()
{
  const std::size_t M = globals::lattice->num_basis_sites();
  const std::size_t Ns = static_cast<std::size_t>(num_periodogram_samples());
  const std::size_t ov = static_cast<std::size_t>(periodogram_overlap());

  if (Ns == 0)
  {
    return;
  }

  assert(ov < Ns);

  const std::size_t src0 = Ns - ov;

  for (std::size_t m = 0; m < M; ++m)
  {
    auto* dst = &sublattice_magnetisation_(m, 0);
    const auto* src = &sublattice_magnetisation_(m, src0);
    std::copy_n(src, ov, dst);
    std::fill_n(&sublattice_magnetisation_(m, ov), Ns - ov, Vec3{0, 0, 0});
  }
}

jams::MultiArray<Vec3, 1> SpectrumBaseMonitor::generate_sublattice_magnetisation_directions_()
{
  if (sw_window_.empty())
  {
    sw_window_ = generate_normalised_window_(num_periodogram_samples());
  }

  jams::MultiArray<Vec3, 1> mean_directions(num_motif_atoms());
  zero(mean_directions);

  for (auto m = 0; m < num_motif_atoms(); ++m)
  {
    for (auto n = 0; n < num_periodogram_samples(); ++n)
    {
      mean_directions(m) += sw_window_(n) * sublattice_magnetisation_(m, n);
    }

    mean_directions(m) = normalize(mean_directions(m));
  }

  return mean_directions;
}

jams::MultiArray<Mat3, 1> SpectrumBaseMonitor::generate_sublattice_rotations_()
{
  jams::MultiArray<Mat3, 1> rotations(num_motif_atoms());
  for (auto a = 0; a < num_motif_atoms(); ++a)
  {
    rotations(a) = kIdentityMat3;
  }

  if (!channel_map_.rotate_to_sublattice_frame)
  {
    return rotations;
  }

  const auto mean_directions = generate_sublattice_magnetisation_directions_();

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

  if (channel_map_.scale_by_spin_length)
  {
    const double mu = globals::mus(basis_index);
    const double spin_length = mu / kElectronGFactor;
    s *= spin_length;
  }

  const auto& w = channel_map_.coeffs[channel_index];
  return w[0] * s[0] + w[1] * s[1] + w[2] * s[2];
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

void SpectrumBaseMonitor::fourier_transform_to_kspace_and_store(const jams::MultiArray<double, 2> &data)
{
  fft_supercell_vector_field_to_kspace(
      data,
      kspace_data_,
      globals::lattice->size(),
      globals::lattice->kspace_size(),
      globals::lattice->num_basis_sites());

  store_sublattice_magnetisation_(data);
  store_kspace_data_on_path(kspace_data_, kspace_paths_);
  periodogram_index_++;
}

void SpectrumBaseMonitor::print_info() const
{
  std::cout << "\n";
  std::cout << "  number of samples " << num_periodogram_samples() << "\n";
  std::cout << "  sampling time (s) " << sample_time_interval() << "\n";
  std::cout << "  acquisition time (s) " << sample_time_interval() * num_periodogram_samples() << "\n";
  std::cout << "  frequency resolution (THz) " << frequency_resolution_thz() << "\n";
  std::cout << "  maximum frequency (THz) " << max_frequency_thz() << "\n";
  std::cout << "  channels " << num_channels() << "\n";
  std::cout << "\n";
}
