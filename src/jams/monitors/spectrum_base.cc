//
// Created by Joseph Barker on 2019-08-01.
//

#include "jams/interface/fft.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/monitors/spectrum_base.h"

#include <iostream>
#include <cmath>

SpectrumBaseMonitor::SpectrumBaseMonitor(const libconfig::Setting &settings) : Monitor(settings) {

  keep_negative_frequencies_ = jams::config_optional<bool>(settings, "keep_negative_frequencies", keep_negative_frequencies_);

  configure_kspace_paths(settings["hkl_path"]);

  if (settings.exists("compute_periodogram")) {
    configure_periodogram(settings["compute_periodogram"]);
  }

  auto kspace_size = globals::lattice->kspace_size();

  num_motif_atoms_ = globals::lattice->num_basis_sites();

  zero(kspace_data_.resize(
      kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_motif_atoms_));
  zero(sk_timeseries_.resize(
      num_motif_atoms_, periodogram_props_.length, kspace_paths_.size()));

  
  skw_spectrum_.resize(num_motif_atoms(), num_periodogram_samples(), num_kpoints());

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

void SpectrumBaseMonitor::insert_full_kspace(Vec3i kspace_size) {
  std::vector<jams::HKLIndex> hkl_path;

  for (auto l = 0; l < kspace_size[0]; ++l) {
    for (auto m = 0; m < kspace_size[1]; ++m) {
      for (auto n = 0; n < kspace_size[2]; ++n) {
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
  kspace_continuous_path_ranges_.push_back(kspace_continuous_path_ranges_.back() + hkl_path.size());
}

void SpectrumBaseMonitor::insert_continuous_kpath(libconfig::Setting& settings) {
  if (!settings.isList())
  {
    throw std::runtime_error("SpectrumBaseMonitor::configure_continuous_kpath failed because settings is not a List");
  }

  std::vector<Vec3> hkl_path_nodes(settings.getLength());
  for (auto i = 0; i < settings.getLength(); ++i) {
    if (!settings[i].isArray())
    {
      throw std::runtime_error("SpectrumBaseMonitor::configure_continuous_kpath failed hkl node is not an Array");
    }

    hkl_path_nodes[i] = Vec3{settings[i][0], settings[i][1], settings[i][2]};
  }

  // If the user gives two nodes which are identical then there is no path
  // (it has no length) which could cause some nasty problems when we try
  // to generate the paths.
  for (auto i = 1; i < hkl_path_nodes.size(); ++i) {
    if (hkl_path_nodes[i] == hkl_path_nodes[i-1]) {
      throw std::runtime_error("Two consecutive hkl_nodes cannot be the same");
    }
  }

  auto new_path = generate_hkl_kspace_path(hkl_path_nodes, globals::lattice->kspace_size());
  kspace_paths_.insert(end(kspace_paths_), begin(new_path), end(new_path));

  if (kspace_continuous_path_ranges_.empty())
  {
    kspace_continuous_path_ranges_.push_back(0);
  }
  kspace_continuous_path_ranges_.push_back(kspace_continuous_path_ranges_.back() + new_path.size());
}

void SpectrumBaseMonitor::configure_kspace_paths(libconfig::Setting& settings) {
  // hkl_path can be a simple list of nodes e.g.
  //     hkl_path = ( [3.0, 3.0,-3.0], [ 5.0, 5.0,-5.0] );
  // or a list of discontinuous paths e.g.
  //    hkl_path = ( ([3.0, 3.0,-3.0], [ 5.0, 5.0,-5.0]),
  //                 ([3.0, 3.0,-2.0], [ 5.0, 5.0,-4.0]));


  // Configur full kspace
  if (settings.isString() && std::string(settings.c_str()) == "full")
  {
    insert_full_kspace(globals::lattice->kspace_size());
    return;
  }

  // Configure a single kpath
  if (settings[0].isArray())
  {
    insert_continuous_kpath(settings);
    return;
  }

  // Configure a list of discontinuous kpaths
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

void SpectrumBaseMonitor::configure_periodogram(libconfig::Setting &settings) {
  periodogram_props_.length = jams::config_required<int>(settings, "length");
  periodogram_props_.overlap = jams::config_optional<int>(settings, "overlap", periodogram_props_.length / 2);

  if (periodogram_props_.length <= 0) {
    throw std::runtime_error("Periodogram length must be greater than zero");
  }

  if (periodogram_props_.overlap <= 0) {
    throw std::runtime_error("Periodogram overlap must be greater than zero");
  }

  if (periodogram_props_.overlap >= periodogram_props_.length) {
    throw std::runtime_error("Periodogram overlap must be less than periodogram length");
  }
}


/**
 * Generate a path between nodes in reciprocal space sampling the kspace discretely.
 *
 * @param hkl_nodes
 * @param kspace_size
 * @return
 */
std::vector<jams::HKLIndex> SpectrumBaseMonitor::generate_hkl_kspace_path(const std::vector<Vec3> &hkl_nodes, const Vec3i &kspace_size) {
  std::vector<jams::HKLIndex> hkl_path;
  // Our sampling of k-space is discrete, with a point per unit cell in the supercell. Therefore the path between
  // nodes must be rasterized onto the discrete grid. Here we use a 3D version of Bresenham's line algorithm to do this.
  // This enables us to have good sampling, even when the line does not exactly hit the discrete k-points, but also
  // avoiding any interpolation which could distort the data. The algorithm works by first identifying the 'driving'
  // axis (x, y or z), the one with the largest displacement and then whilst stepping along that axis, adjusting the
  // steps along the remaining axes.

  for (auto n = 0; n < hkl_nodes.size()-1; ++n) {
    Vec3i start = to_int(hadamard_product(hkl_nodes[n], kspace_size));
    Vec3i end = to_int(hadamard_product(hkl_nodes[n + 1], kspace_size));
    Vec3i displacement = absolute(end - start);

    Vec3i step = {
        (end[0] > start[0]) ? 1 : -1,
        (end[1] > start[1]) ? 1 : -1,
        (end[2] > start[2]) ? 1 : -1};

    // x-axis is driving axis
    if (displacement[0] >= displacement[1] && displacement[0] >= displacement[2]) {
      int p1 = 2 * displacement[1] - displacement[0];
      int p2 = 2 * displacement[2] - displacement[0];
      while (start[0] != end[0]) {
        Vec3 hkl = hadamard_product(start, 1.0 / to_double(kspace_size));
        Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
        hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[0] += step[0];
        if (p1 >= 0) {
          start[1] += step[1];
          p1 -= 2 * displacement[0];
        }
        if (p2 >= 0) {
          start[2] += step[2];
          p2 -= 2 * displacement[0];
        }
        p1 += 2 * displacement[1];
        p2 += 2 * displacement[2];

      }
    }
    // y-axis is driving axis
    else if (displacement[1] >= displacement[0] && displacement[1] >= displacement[2]) {
      int p1 = 2 * displacement[0] - displacement[1];
      int p2 = 2 * displacement[2] - displacement[1];
      while (start[1] != end[1]) {
        Vec3 hkl = hadamard_product(start, 1.0 / to_double(kspace_size));
        Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
        hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[1] += step[1];
        if (p1 >= 0) {
          start[0] += step[0];
          p1 -= 2 * displacement[1];
        }
        if (p2 >= 0) {
          start[2] += step[2];
          p2 -= 2 * displacement[1];
        }
        p1 += 2 * displacement[0];
        p2 += 2 * displacement[2];

      }
    }
    // z-axis is driving axis
    else {
      int p1 = 2 * displacement[0] - displacement[2];
      int p2 = 2 * displacement[1] - displacement[2];
      while (start[2] != end[2]) {
        Vec3 hkl = hadamard_product(start, 1.0 / to_double(kspace_size));
        Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
        hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[2] += step[2];
        if (p1 >= 0) {
          start[1] += step[1];
          p1 -= 2 * displacement[2];
        }
        if (p2 >= 0) {
          start[0] += step[0];
          p2 -= 2 * displacement[2];
        }
        p1 += 2 * displacement[1];
        p2 += 2 * displacement[0];

      }
    }

    //include final point
    Vec3 hkl = hadamard_product(end, 1.0 / to_double(kspace_size));
    Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
    hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});
  }

  // remove duplicates in the path where start and end indicies are the same at nodes
  hkl_path.erase(unique(hkl_path.begin(), hkl_path.end()), hkl_path.end());

  return hkl_path;
}

jams::MultiArray<Vec3cx,2>& SpectrumBaseMonitor::fft_sk_timeseries_to_skw(
  const int kpoint_index,
  const CmplxVecField& timeseries
  )
{
  const int num_sites         = timeseries.size(0);
  const int num_time_samples  = timeseries.size(1);

  // We need to resize and re-plan the fft if sw_spectrum_buffer_ or sw_spectrum_fft_plan_
  // are uninitialised or sw_spectrum_buffer_ has changed size
  if (
    sw_window_.size() != num_time_samples
    || sw_spectrum_buffer_.size(0) != num_sites
    || sw_spectrum_buffer_.size(1) != num_time_samples
    || !sw_spectrum_fft_plan_)
  {

    sw_window_ = generate_normalised_window_(num_time_samples);

    sw_spectrum_buffer_.resize(num_sites, num_time_samples);

    // Time FFT per (site, k-point). We transform 3 complex channels (Vec3cx) for each fixed (a,k)
    // along the time dimension.
    // Memory layout assumption for jams::MultiArray<Vec3cx,3> with indices (site, time, k):
    // k is the fastest index, then time, then site. Each Vec3cx stores 3 complex values contiguously.

    const int n[1] = { num_time_samples };
    const int howmany = 3; // x/y/z or +/−/z channels

    // Stride between successive time samples for a fixed (a,k), in units of fftw_complex.
    // Advancing time by 1 jumps over all k-points worth of Vec3cx (each Vec3cx has 3 complex entries).
    const int istride = howmany;
    const int ostride = howmany;

    // Distance between the start of each transform (channel) for a fixed (a,k), in units of fftw_complex.
    // The 3 channels are stored contiguously within Vec3cx.
    const int idist = 1;
    const int odist = 1;

    // Create a plan once and re-use it for all (a,k) by calling fftw_execute_dft with different pointers.
    // Use a non-null dummy pointer to satisfy FFTW plan creation requirements.
    auto* dummy = FFTW_COMPLEX_CAST(&sw_spectrum_buffer_(0, 0));

    sw_spectrum_fft_plan_ = fftw_plan_many_dft(
        /*rank=*/1,
        /*n=*/n,
        /*howmany=*/howmany,
        /*in=*/dummy,
        /*inembed=*/nullptr,
        /*istride=*/istride,
        /*idist=*/idist,
        /*out=*/dummy,
        /*onembed=*/nullptr,
        /*ostride=*/ostride,
        /*odist=*/odist,
        /*sign=*/FFTW_FORWARD,
        /*flags=*/FFTW_ESTIMATE);

    assert(sw_spectrum_fft_plan_);
  }


  if (channel_mapping_ == ChannelMapping::Cartesian)
  {
    for (auto a = 0; a < num_sites; ++a)
    {

      for (auto t = 0; t < num_time_samples; ++t) {
        sw_spectrum_buffer_(a, t) = timeseries(a, t, kpoint_index);
      }
    }
  }

  if (channel_mapping_ == ChannelMapping::RaiseLower)
  {
    auto rotations = generate_sublattice_rotations_();
    for (auto a = 0; a < num_motif_atoms(); ++a)
    {
      const double mu        = globals::mus(a);                 // in mu_B
      const double Slen      = mu / kElectronGFactor; // dimensionless spin length

      for (auto t = 0; t < num_time_samples; ++t) {
        auto x = timeseries(a, t, kpoint_index);
        // Rotate unit spin into local frame, then scale by spin length to get physical components.
        x = rotations(a) * x;
        x *= Slen;

        // Map (Sx,Sy,Sz) -> (S+, S-, Sz) with unitary 1/sqrt(2).
        sw_spectrum_buffer_(a, t) = Vec3cx{
          kInvSqrtTwo * (x[0] + kImagOne * x[1]),
          kInvSqrtTwo * (x[0] - kImagOne * x[1]),
          x[2]
        };
      }
    }
  }

  const double time_norm = 1.0 / static_cast<double>(num_time_samples);

  // Remove the static component from the time series and apply the window function
  for (auto a = 0; a < num_sites; ++a)
  {
    Vec3cx sk0{};
    for (auto t = 0; t < num_time_samples; ++t) {
      sk0 += time_norm * sw_spectrum_buffer_(a, t);
    }

    for (auto t = 0; t < num_time_samples; ++t) {
        sw_spectrum_buffer_(a, t) = time_norm * sw_window_(t) * (sw_spectrum_buffer_(a, t) - sk0);
    }
  }

  for (int a = 0; a < num_sites; ++a)
  {
      auto* ptr = FFTW_COMPLEX_CAST(&sw_spectrum_buffer_(a, 0));
      fftw_execute_dft(sw_spectrum_fft_plan_, ptr, ptr);
  }

  return sw_spectrum_buffer_;
}

SpectrumBaseMonitor::CmplxVecField& SpectrumBaseMonitor::fft_timeseries_to_frequency(const CmplxVecField& timeseries) {
  // Make a local working copy for mean subtraction, windowing, and in-place FFT.
  CmplxVecField spectrum = timeseries;

  const int num_sites         = spectrum.size(0);
  const int num_time_samples  = spectrum.size(1);
  const int num_space_samples = spectrum.size(2);

  // Normalise the FFT window to unit RMS power so that windowing does not change overall power.
  // For Welch/periodogram-style spectra, this makes amplitudes comparable across window choices.
  double w2sum = 0.0;
  jams::MultiArray<double, 1> window(num_time_samples);
  for (auto i = 0; i < num_time_samples; ++i) {
    window(i) = fft_window_default(i, num_time_samples);
    w2sum += window(i) * window(i);
  }
  const double w_rms = (w2sum > 0.0) ? std::sqrt(w2sum / double(num_time_samples)) : 1.0;
  const double win_norm = (w_rms > 0.0) ? (1.0 / w_rms) : 1.0;
  const double time_norm = 1.0 / static_cast<double>(num_time_samples);

  jams::MultiArray<Vec3cx, 1> sk0(num_space_samples);

  for (auto a = 0; a < num_sites; ++a)
  {
    zero(sk0);
    for (auto t = 0; t < num_time_samples; ++t) {
      for (auto k = 0; k < num_space_samples; ++k) {
        sk0(k) += time_norm * spectrum(a, t, k);
      }
    }

    for (auto t = 0; t < num_time_samples; ++t) {
      for (auto k = 0; k < num_space_samples; ++k) {
        spectrum(a, t, k) = time_norm * win_norm * window(t) * (spectrum(a, t, k) - sk0(k));
      }
    }
  }

  // Time FFT per (site, k-point). We transform 3 complex channels (Vec3cx) for each fixed (a,k)
  // along the time dimension.
  // Memory layout assumption for jams::MultiArray<Vec3cx,3> with indices (site, time, k):
  // k is the fastest index, then time, then site. Each Vec3cx stores 3 complex values contiguously.

  const int n[1] = { num_time_samples };
  const int howmany = 3; // x/y/z or +/−/z channels

  // Stride between successive time samples for a fixed (a,k), in units of fftw_complex.
  // Advancing time by 1 jumps over all k-points worth of Vec3cx (each Vec3cx has 3 complex entries).
  const int istride = num_space_samples * howmany;
  const int ostride = num_space_samples * howmany;

  // Distance between the start of each transform (channel) for a fixed (a,k), in units of fftw_complex.
  // The 3 channels are stored contiguously within Vec3cx.
  const int idist = 1;
  const int odist = 1;

  // Create a plan once and re-use it for all (a,k) by calling fftw_execute_dft with different pointers.
  // Use a non-null dummy pointer to satisfy FFTW plan creation requirements.
  auto* dummy = FFTW_COMPLEX_CAST(&spectrum(0, 0, 0));

  fftw_plan plan = fftw_plan_many_dft(
      /*rank=*/1,
      /*n=*/n,
      /*howmany=*/howmany,
      /*in=*/dummy,
      /*inembed=*/nullptr,
      /*istride=*/istride,
      /*idist=*/idist,
      /*out=*/dummy,
      /*onembed=*/nullptr,
      /*ostride=*/ostride,
      /*odist=*/odist,
      /*sign=*/FFTW_FORWARD,
      /*flags=*/FFTW_ESTIMATE);

  assert(plan);

  for (int a = 0; a < num_sites; ++a)
  {
    for (int k = 0; k < num_space_samples; ++k)
    {
      auto* ptr = FFTW_COMPLEX_CAST(&spectrum(a, 0, k));
      fftw_execute_dft(plan, ptr, ptr);
    }
  }

  fftw_destroy_plan(plan);

  const int num_freq_out = keep_negative_frequencies_
                           ? num_time_samples
                           : (num_time_samples / 2 + 1);

  if (skw_spectrum_.size(0) != num_sites ||
      skw_spectrum_.size(1) != num_freq_out ||
      skw_spectrum_.size(2) != num_space_samples)
  {
    skw_spectrum_.resize(num_sites, num_freq_out, num_space_samples);
  }

  for (int a = 0; a < num_sites; ++a)
  {
    for (int f = 0; f < num_freq_out; ++f)
    {
      for (int k = 0; k < num_space_samples; ++k)
      {
        skw_spectrum_(a, f, k) = spectrum(a, f, k);
      }
    }
  }

  return skw_spectrum_;
}

bool SpectrumBaseMonitor::do_periodogram_update() const {
  return periodogram_index_ >= periodogram_props_.length && periodogram_props_.length > 0;
}

void SpectrumBaseMonitor::shift_periodogram()
{
  shift_sk_timeseries_();
  shift_sublattice_magnetisation_timeseries_();
  periodogram_index_ = periodogram_props_.overlap;
  total_periods_++;
}

SpectrumBaseMonitor::CmplxVecField SpectrumBaseMonitor::compute_periodogram_spectrum(CmplxVecField &timeseries) {
  auto& spectrum = fft_timeseries_to_frequency(timeseries);
  shift_sk_timeseries_();
  periodogram_index_ = periodogram_props_.overlap;
  total_periods_++;
  return spectrum;
}


SpectrumBaseMonitor::CmplxVecField SpectrumBaseMonitor::compute_periodogram_rotated_spectrum(
    CmplxVecField &timeseries)
{

  auto sublattice_rotations = generate_sublattice_rotations_();
  auto sublattice_channel_mappings = generate_sublattice_channel_mappings_();

  for (auto k = 0; k < timeseries.size(2); ++k)
  {
    auto sw_spectrum = fft_sk_timeseries_to_skw(k, timeseries);
    for (auto a = 0; a < timeseries.size(0); ++a)
    {
      for (auto f = 0; f < sw_spectrum.size(1); ++f)
      {
        skw_spectrum_(a, f, k) = sw_spectrum(a, f);
      }
    }
  }

  // auto& spectrum = fft_timeseries_to_frequency(transformed_timeseries);

  shift_sk_timeseries_();
  periodogram_index_ = periodogram_props_.overlap;
  total_periods_++;

  return skw_spectrum_;
}

// void SpectrumBaseMonitor::shift_sk_timeseries_(CmplxVecField &timeseries, int overlap) {
//   assert(overlap < timeseries.size(1));
//   // shift overlap data to the start of the range
//   for (auto a = 0; a < timeseries.size(0); ++a) {           // motif atom
//     for (auto i = 0; i < overlap; ++i) {    // time index
//       for (auto j = 0; j < timeseries.size(2); ++j) {       // kpoint index
//         timeseries(a, i, j) = timeseries(a, timeseries.size(1) - overlap + i, j);
//       }
//     }
//   }
// }

void SpectrumBaseMonitor::shift_sk_timeseries_()
{
  const std::size_t A = sk_timeseries_.size(0);
  const std::size_t T = sk_timeseries_.size(1);
  const std::size_t K = sk_timeseries_.size(2);

  const std::size_t ov = periodogram_overlap();
  assert(ov < T);

  const std::size_t src0 = T - ov;

  for (std::size_t a = 0; a < A; ++a)
  {
    for (std::size_t i = 0; i < ov; ++i)
    {
      auto*       dst = &sk_timeseries_(a, i, 0);
      const auto* src = &sk_timeseries_(a, src0 + i, 0);
      std::copy_n(src, K, dst);
    }
  }
}

void SpectrumBaseMonitor::store_kspace_data_on_path(const jams::MultiArray<Vec3cx,4> &kspace_data, const std::vector<jams::HKLIndex> &kspace_path) {
  for (auto a = 0; a < kspace_data.size(3); ++a) {
    for (auto k = 0; k < kspace_path.size(); ++k) {
      auto kindex = kspace_path[k].index;
      auto i = periodogram_index_;
      auto idx = kindex.offset;
      if (kindex.conj) {
        sk_timeseries_(a, i, k) = sk_phase_factors_(a, k) * conj(kspace_data(idx[0], idx[1], idx[2], a));
      } else {
        sk_timeseries_(a, i, k) = sk_phase_factors_(a, k) * kspace_data(idx[0], idx[1], idx[2], a);
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
    Vec3 spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    auto m = globals::lattice->lattice_site_basis_index(i);
    sublattice_magnetisation_(m, p) += spin;
  }
}

void SpectrumBaseMonitor::shift_sublattice_magnetisation_timeseries_()
{
  const std::size_t M  = globals::lattice->num_basis_sites();
  const std::size_t Ns = static_cast<std::size_t>(num_periodogram_samples());
  const std::size_t ov = periodogram_overlap();

  assert(ov < Ns);

  const std::size_t src0 = Ns - ov;

  for (std::size_t m = 0; m < M; ++m)
  {
    // Copy overlap block to the start: [src0, Ns) -> [0, ov)
    auto*       dst = &sublattice_magnetisation_(m, 0);
    const auto* src = &sublattice_magnetisation_(m, src0);
    std::copy_n(src, ov, dst);

    // Zero the tail: [ov, Ns)
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

  // Calculate the mean magnetisation direction for each sublattice across this periodogram period.
  // We use the same windowing function as the FFT for consistency.
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
  if (channel_mapping_ == ChannelMapping::Cartesian)
  {
    jams::MultiArray<Mat3, 1> rotations(num_motif_atoms());
    for (auto a = 0; a < num_motif_atoms(); ++a)
    {
      rotations(a) = kIdentityMat3;
    }
    return rotations;
  }

  const auto mean_directions = generate_sublattice_magnetisation_directions_();

  jams::MultiArray<Mat3, 1> rotations(mean_directions.size());
        for (auto m = 0; m < mean_directions.size(); ++m)
        {
            // Construct a local transverse basis (e1,e2,n) with a fixed gauge.
            // This avoids the ill-conditioning of "minimal rotation" when n ≈ z.
            Vec3 n_hat = mean_directions(m);
            const double n_norm = norm(n_hat);
            if (n_norm <= 0.0)
            {
                rotations(m) = kIdentityMat3;
                continue;
            }
            n_hat *= (1.0 / n_norm);

            // Choose the global Cartesian axis least aligned with n_hat to avoid discontinuous gauge flips.
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

            // e1 = normalised projection of r into the plane normal to n
            Vec3 e1 = r - dot(r, n_hat) * n_hat;
            const double e1_norm = norm(e1);
            if (e1_norm <= 0.0)
            {
                // Extremely unlikely after the fallback, but be safe.
                rotations(m) = kIdentityMat3;
                continue;
            }
            e1 *= (1.0 / e1_norm);

            // e2 completes a right-handed orthonormal triad
            Vec3 e2 = cross(n_hat, e1);

            // Build rotation matrix that maps global -> local components:
            // v_local = [e1^T; e2^T; n^T] v_global.
            Mat3 R = kIdentityMat3;
            R[0][0] = e1[0];    R[0][1] = e1[1];    R[0][2] = e1[2];
            R[1][0] = e2[0];    R[1][1] = e2[1];    R[1][2] = e2[2];
            R[2][0] = n_hat[0]; R[2][1] = n_hat[1]; R[2][2] = n_hat[2];

            rotations(m) = R;
        }
  return rotations;
}

jams::MultiArray<Mat3cx, 1> SpectrumBaseMonitor::generate_sublattice_channel_mappings_()
{
  jams::MultiArray<Mat3cx, 1> channel_mappings(num_motif_atoms());

  if (channel_mapping_ == ChannelMapping::Cartesian)
  {
    for (auto a = 0; a < num_motif_atoms(); ++a)
    {
      channel_mappings(a) = kIdentityMat3cx;
    }
  }

  for (auto a = 0; a < num_motif_atoms(); ++a)
  {
    const double mu   = globals::mus(a); // in mu_B
    const double Slen = mu / kElectronGFactor;

    // First scale (Sx,Sy,Sz) by spin length Slen, then map to (S+,S-,Sz).
    channel_mappings(a) = Mat3cx{
      {
        Slen * kInvSqrtTwo, +Slen * kImagOne * kInvSqrtTwo, 0.0,
        Slen * kInvSqrtTwo, -Slen * kImagOne * kInvSqrtTwo, 0.0,
        0.0,              0.0,                         Slen
      }};
  }
  return channel_mappings;
}

jams::MultiArray<double, 1>
SpectrumBaseMonitor::generate_normalised_window_(int num_time_samples)
{
  jams::MultiArray<double, 1> window(num_time_samples);

  double w2sum = 0.0;
  for (int i = 0; i < num_time_samples; ++i) {
    const double w = fft_window_default(i, num_time_samples);
    window(i) = w;
    w2sum += w * w;
  }

  const double inv_rms =
      (w2sum > 0.0) ? 1.0 / std::sqrt(w2sum / static_cast<double>(num_time_samples))
                    : 1.0;

  for (int i = 0; i < num_time_samples; ++i) {
    window(i) *= inv_rms;
  }

  return window;
}

jams::MultiArray<jams::ComplexHi, 2> SpectrumBaseMonitor::generate_phase_factors_(const std::vector<Vec3>& r_frac,
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

void SpectrumBaseMonitor::fourier_transform_to_kspace_and_store(const jams::MultiArray<double, 2> &data) {
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

void SpectrumBaseMonitor::apply_raise_lower_mapping(CmplxVecField& timeseries, const jams::MultiArray<Mat3, 1>& rotations)
{
  for (auto m = 0; m < timeseries.size(0); ++m) // motif/basis site
  {
    const double mu   = globals::mus(m); // in mu_B
    const double Slen = mu / kElectronGFactor;
    const auto   R    = rotations(m);
    for (auto i = 0; i < timeseries.size(1); ++i) // periodogram_index
    {
      for (auto n = 0; n < timeseries.size(2); ++n) // kpath_index
      {
        auto rotated_spin = Slen * (R * timeseries(m, i, n));

        Vec3cx remapped_spin = {
          kInvSqrtTwo * (rotated_spin[0] + kImagOne * rotated_spin[1]),
          kInvSqrtTwo * (rotated_spin[0] - kImagOne * rotated_spin[1]),
          rotated_spin[2] };

        timeseries(m, i, n) = remapped_spin;
      }
    }
  }
}

void SpectrumBaseMonitor::print_info() const {
  std::cout << "\n";
  std::cout << "  number of samples "          << num_periodogram_samples() << "\n";
  std::cout << "  sampling time (s) "          << sample_time_interval() << "\n";
  std::cout << "  acquisition time (s) "       << sample_time_interval() * num_periodogram_samples() << "\n";
  std::cout << "  frequency resolution (THz) " << frequency_resolution_thz() << "\n";
  std::cout << "  maximum frequency (THz) "    << max_frequency_thz() << "\n";
  std::cout << "\n";
}
