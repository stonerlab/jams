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
  periodogram_props_ = {0, 0};
  configure_kspace_paths(settings["hkl_path"]);

  if (settings.exists("compute_periodogram")) {
    configure_periodogram(settings["compute_periodogram"]);
  }

  auto kspace_size   = globals::lattice->kspace_size();
  num_motif_atoms_ = globals::lattice->num_basis_sites();

  zero(kspace_data_.resize(
      kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_motif_atoms_));
  zero(kspace_data_timeseries_.resize(
      num_motif_atoms_, periodogram_props_.length, kspace_paths_.size()));
}


void SpectrumBaseMonitor::configure_kspace_paths(libconfig::Setting& settings) {
  // hkl_path can be a simple list of nodes e.g.
  //     hkl_path = ( [3.0, 3.0,-3.0], [ 5.0, 5.0,-5.0] );
  // or a list of discontinuous paths e.g.
  //    hkl_path = ( ([3.0, 3.0,-3.0], [ 5.0, 5.0,-5.0]),
  //                 ([3.0, 3.0,-2.0], [ 5.0, 5.0,-4.0]));

  if (!(settings[0].isList() || settings[0].isArray())) {
    throw std::runtime_error("NeutronScatteringMonitor:: hkl_nodes must be a list or a group");
  }

  bool has_discontinuous_paths = settings[0].isList();

  kspace_continuous_path_ranges_.push_back(0);
  if (has_discontinuous_paths) {
    for (auto n = 0; n < settings.getLength(); ++n) {
      std::vector<Vec3> hkl_path_nodes(settings[n].getLength());
      for (auto i = 0; i < settings[n].getLength(); ++i) {
        hkl_path_nodes[i] = Vec3{settings[n][i][0], settings[n][i][1], settings[n][i][2]};
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
      kspace_continuous_path_ranges_.push_back(kspace_continuous_path_ranges_.back() + new_path.size());
    }
  } else {
    std::vector<Vec3> hkl_path_nodes(settings.getLength());
    for (auto i = 0; i < settings.getLength(); ++i) {
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

    kspace_paths_ = generate_hkl_kspace_path(hkl_path_nodes, globals::lattice->kspace_size());
    kspace_continuous_path_ranges_.push_back(kspace_continuous_path_ranges_.back() + kspace_paths_.size());
  }
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

  int rank = 1;
  int transform_size[1] = {num_time_samples};
  int num_transforms = num_space_samples * 3;
  int nembed[1] = {num_time_samples};
  int stride = num_space_samples * 3;
  int dist = 1;

  // Normalise the FFT window to unit RMS power so that windowing does not change overall power.
  // For Welch/periodogram-style spectra, this makes amplitudes comparable across window choices.
  double w2sum = 0.0;
  for (auto i = 0; i < num_time_samples; ++i) {
    const double w = fft_window_default(i, num_time_samples);
    w2sum += w * w;
  }
  const double w_rms = (w2sum > 0.0) ? std::sqrt(w2sum / double(num_time_samples)) : 1.0;
  const double win_norm = (w_rms > 0.0) ? (1.0 / w_rms) : 1.0;

  for (auto a = 0; a < num_sites; ++a) {
    fftw_plan fft_plan = fftw_plan_many_dft(rank, transform_size, num_transforms,
                                            FFTW_COMPLEX_CAST(&spectrum(a,0,0)), nembed, stride, dist,
                                            FFTW_COMPLEX_CAST(&spectrum(a,0,0)), nembed, stride, dist,
                                            FFTW_FORWARD, FFTW_ESTIMATE);

    assert(fft_plan);

    jams::MultiArray<Vec3cx, 1> static_spectrum(num_space_samples);
    zero(static_spectrum);
    for (auto i = 0; i < num_time_samples; ++i) {
      for (auto j = 0; j < num_space_samples; ++j) {
        static_spectrum(j) += spectrum(a, i, j);
      }
    }
    element_scale(static_spectrum, 1.0/double(num_time_samples));

    for (auto i = 0; i < num_time_samples; ++i) {
      for (auto j = 0; j < num_space_samples; ++j) {
        spectrum(a, i, j) = (win_norm * fft_window_default(i, num_time_samples)) * (spectrum(a, i, j) - static_spectrum(j));
      }
    }

    fftw_execute(fft_plan);
    fftw_destroy_plan(fft_plan);
  }
  element_scale(spectrum, 1.0 / double(num_time_samples));

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

  auto transformed_timeseries = rotate_sk_timeseries(timeseries, rotations);
  transformed_timeseries = apply_sk_channel_mapping(transformed_timeseries);


  auto spectrum = fft_timeseries_to_frequency(transformed_timeseries);
  shift_periodogram_timeseries(timeseries, periodogram_props_.overlap);
  // put the pointer to the overlap position
  periodogram_index_ = periodogram_props_.overlap;
  total_periods_++;

  return spectrum;
}

void SpectrumBaseMonitor::shift_periodogram_timeseries(CmplxVecField &timeseries, int overlap) {
  assert(overlap < timeseries.size(1));
  // shift overlap data to the start of the range
  for (auto a = 0; a < timeseries.size(0); ++a) {           // motif atom
    for (auto i = 0; i < overlap; ++i) {    // time index
      for (auto j = 0; j < timeseries.size(2); ++j) {       // kpoint index
        timeseries(a, i, j) = timeseries(a, timeseries.size(1) - overlap + i, j);
      }
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
        kspace_data_timeseries_(a, i, k) = conj(kspace_data(idx[0], idx[1], idx[2], a));
      } else {
        kspace_data_timeseries_(a, i, k) = kspace_data(idx[0], idx[1], idx[2], a);
      }
    }
  }
}

void SpectrumBaseMonitor::store_periodogram_data(const jams::MultiArray<double, 2> &data) {
  fft_supercell_vector_field_to_kspace(data, kspace_data_, globals::lattice->size(), globals::lattice->kspace_size(),
                                       globals::lattice->num_basis_sites());
  store_kspace_data_on_path(kspace_data_, kspace_paths_);
  periodogram_index_++;
}

SpectrumBaseMonitor::CmplxVecField SpectrumBaseMonitor::rotate_sk_timeseries(const CmplxVecField& timeseries,
  const jams::MultiArray<Mat3, 1>& rotations)
{
  auto rotated_timeseries = timeseries;

  for (auto m = 0; m < rotated_timeseries.size(0); ++m) // num_sites
  {
    const auto R = rotations(m); // maps mean direction to +z (see construction above)
    for (auto i = 0; i < rotated_timeseries.size(1); ++i) // periodogram_index
    {
      for (auto n = 0; n < rotated_timeseries.size(2); ++n) // kpath_index
      {
        rotated_timeseries(m, i, n) = R * rotated_timeseries(m, i, n);
      }
    }
  }
  return rotated_timeseries;
}

SpectrumBaseMonitor::CmplxVecField SpectrumBaseMonitor::apply_sk_channel_mapping(const CmplxVecField& timeseries)
{
  auto mapped_timeseries = timeseries;
  for (auto m = 0; m < mapped_timeseries.size(0); ++m) // num_sites
  {
    for (auto i = 0; i < timeseries.size(1); ++i) // periodogram_index
    {
      for (auto n = 0; n < timeseries.size(2); ++n) // kpath_index
      {
        mapped_timeseries(m, i, n) = channel_mapping_ * mapped_timeseries(m, i, n);

        if (do_boson_normalisation_) {
          // After channel mapping, channels are (+, -, z).
          // Convert to approximate boson amplitudes a ~ S^+ / sqrt(2S).
          // globals::s stores unit spins; globals::mus stores moments in mu_B.
          // S = mu/(g mu_B) meaning a ~ = (s^+ * mu/(g mu_B)) / sqrt(2 * mu/(g mu_B)) = s^+ sqrt(mu / (2 g mu_B))
          const double mu = globals::mus(m);
          if (mu > 0.0) {
            const double scale = std::sqrt(mu / (2 * kElectronGFactor * kBohrMagnetonIU));
            mapped_timeseries(m, i, n)[0] *= scale; // S+
            mapped_timeseries(m, i, n)[1] *= scale; // S-
            // z unchanged
          }
        }
      }
    }
  }
  return mapped_timeseries;

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
