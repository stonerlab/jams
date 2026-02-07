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

  sk_phase_factors_.resize(num_motif_atoms(), num_kpoints());
  for (auto a = 0; a < num_motif_atoms(); ++a)
  {
    const Vec3 r = globals::lattice->basis_site_atom(a).position_frac;
    for (auto k = 0; k < num_kpoints(); ++k)
    {
      auto q = kspace_paths_[k].hkl;
      sk_phase_factors_(a, k) = exp(-kImagTwoPi * dot(q, r));
    }
  }

}

void SpectrumBaseMonitor::insert_full_kspace(Vec3i kspace_size) {
  assert(do_full_kspace_);
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

jams::MultiArray<Vec3cx,3> SpectrumBaseMonitor::fft_timeseries_to_frequency(jams::MultiArray<Vec3cx, 3> spectrum) {
  // pass spectrum by value to make a copy

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

  const int rank = 1;

  const int num_transforms_per_site   = num_space_samples * 3;   // transforms per site
  const int istride_time = num_transforms_per_site;
  const int ostride_time = num_transforms_per_site;

  const int site_block   = num_time_samples * num_transforms_per_site;

  fftw_iodim dims[rank];
  dims[0].n  = num_time_samples;
  dims[0].is = istride_time;
  dims[0].os = ostride_time;

  // Two howmany dimensions: (tr within site), then (site)
  fftw_iodim howmany_dims[2];

  // Transform index (tr): contiguous
  howmany_dims[0].n  = num_transforms_per_site;
  howmany_dims[0].is = 1;
  howmany_dims[0].os = 1;

  // Site index (a): jump by whole site block
  howmany_dims[1].n  = num_sites;
  howmany_dims[1].is = site_block;
  howmany_dims[1].os = site_block;

  auto* data = FFTW_COMPLEX_CAST(&spectrum(0, 0, 0));

  fftw_plan plan = fftw_plan_guru_dft(
      rank, dims,
      2, howmany_dims,
      data, data,
      FFTW_FORWARD, FFTW_ESTIMATE);

  assert(plan);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  return spectrum;
}

bool SpectrumBaseMonitor::do_periodogram_update() const {
  return periodogram_index_ >= periodogram_props_.length && periodogram_props_.length > 0;
}

SpectrumBaseMonitor::CmplxVecField SpectrumBaseMonitor::compute_periodogram_spectrum(CmplxVecField &timeseries) {
  auto spectrum = fft_timeseries_to_frequency(timeseries);
  shift_periodogram_timeseries(timeseries, periodogram_props_.overlap);
  // put the pointer to the overlap position
  periodogram_index_ = periodogram_props_.overlap;
  total_periods_++;

  return spectrum;
}

SpectrumBaseMonitor::CmplxVecField SpectrumBaseMonitor::compute_periodogram_rotated_spectrum(CmplxVecField &timeseries, const jams::MultiArray<Mat3, 1>& rotations) {
  auto transformed_timeseries = timeseries;
  if (channel_mapping_ == ChannelMapping::RaiseLower)
  {
    apply_raise_lower_mapping(transformed_timeseries, rotations);
  }

  auto spectrum = fft_timeseries_to_frequency(transformed_timeseries);
  shift_periodogram_timeseries(timeseries, periodogram_props_.overlap);
  // put the pointer to the overlap position
  periodogram_index_ = periodogram_props_.overlap;
  total_periods_++;

  return spectrum;
}

// void SpectrumBaseMonitor::shift_periodogram_timeseries(CmplxVecField &timeseries, int overlap) {
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

void SpectrumBaseMonitor::shift_periodogram_timeseries(CmplxVecField& timeseries, int overlap)
{
  const std::size_t A = timeseries.size(0);
  const std::size_t T = timeseries.size(1);
  const std::size_t K = timeseries.size(2);

  assert(overlap >= 0);
  const std::size_t ov = static_cast<std::size_t>(overlap);
  assert(ov < T);

  const std::size_t src0 = T - ov;

  for (std::size_t a = 0; a < A; ++a)
  {
    for (std::size_t i = 0; i < ov; ++i)
    {
      auto*       dst = &timeseries(a, i, 0);
      const auto* src = &timeseries(a, src0 + i, 0);
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

void SpectrumBaseMonitor::fourier_transform_to_kspace_and_store(const jams::MultiArray<double, 2> &data) {
  fft_supercell_vector_field_to_kspace(
    data,
    kspace_data_,
    globals::lattice->size(),
    globals::lattice->kspace_size(),
    globals::lattice->num_basis_sites());

  store_kspace_data_on_path(kspace_data_, kspace_paths_);
  periodogram_index_++;
}

void SpectrumBaseMonitor::apply_raise_lower_mapping(CmplxVecField& timeseries, const jams::MultiArray<Mat3, 1>& rotations)
{
  const double kInvSqrt2 = 1.0/sqrt(2.0);

  for (auto m = 0; m < timeseries.size(0); ++m) // num_sites
  {
    const double S = kElectronGFactor * globals::mus(m);
    const auto R = rotations(m);
    for (auto i = 0; i < timeseries.size(1); ++i) // periodogram_index
    {
      for (auto n = 0; n < timeseries.size(2); ++n) // kpath_index
      {
        auto rotated_spin = S * (R *timeseries(m, i, n));

        Vec3cx remapped_spin = {
          kInvSqrt2 * (rotated_spin[0] + kImagOne * rotated_spin[1]),
          kInvSqrt2 * (rotated_spin[0] - kImagOne * rotated_spin[1]),
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
