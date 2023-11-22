//
// Created by Joseph Barker on 2019-08-01.
//

#include "jams/interface/fft.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/monitors/spectrum_base.h"

#include <iostream>

SpectrumBaseMonitor::SpectrumBaseMonitor(const libconfig::Setting &settings) : Monitor(settings) {
  configure_kspace_paths(settings["hkl_path"]);

  if (settings.exists("compute_periodogram")) {
    configure_periodogram(settings["compute_periodogram"]);
  }

  auto kspace_size   = globals::lattice->kspace_size();
  num_motif_atoms_ = globals::lattice->num_motif_atoms();

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
  periodogram_props_.length = settings["length"];
  periodogram_props_.overlap = settings["overlap"];
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
  for (auto n = 0; n < hkl_nodes.size()-1; ++n) {
    Vec3i origin = to_int(hadamard_product(hkl_nodes[n], kspace_size));
    Vec3i displacement = to_int(hadamard_product(hkl_nodes[n + 1], kspace_size)) - origin;

    const int divisor = gcd( gcd( std::abs(displacement[0]), std::abs(displacement[1]) ), std::abs(displacement[2]) );
    Vec3i delta = displacement / divisor;

    // use +1 to include the last point on the displacement
    const auto num_coordinates = divisor + 1;

    Vec3i coordinate = origin;
    for (auto i = 0; i < num_coordinates; ++i) {
      // map an arbitrary coordinate into the limited k indicies of the reduced brillouin zone
      Vec3 hkl = hadamard_product(coordinate, 1.0 / to_double(kspace_size));
      Vec3 xyz = globals::lattice->get_unitcell().inv_fractional_to_cartesian(hkl);
      hkl_path.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(coordinate, kspace_size)});

      coordinate += delta;
    }
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

  for (auto a = 0; a < num_sites; ++a) {
    fftw_plan fft_plan = fftw_plan_many_dft(rank, transform_size, num_transforms,
                                            FFTW_COMPLEX_CAST(&spectrum(a,0,0)), nembed, stride, dist,
                                            FFTW_COMPLEX_CAST(&spectrum(a,0,0)), nembed, stride, dist,
                                            FFTW_BACKWARD, FFTW_ESTIMATE);

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
        spectrum(a, i, j) = fft_window_default(i, num_time_samples)*(spectrum(a, i, j) - static_spectrum(j));
      }
    }

    fftw_execute(fft_plan);
    fftw_destroy_plan(fft_plan);
  }
  element_scale(spectrum, 1.0 / double(num_time_samples));

  return spectrum;
}

bool SpectrumBaseMonitor::do_periodogram_update() {
  return is_multiple_of(periodogram_index_, periodogram_props_.length);
}

SpectrumBaseMonitor::CmplxVecField SpectrumBaseMonitor::compute_periodogram_spectrum(CmplxVecField &timeseries) {
    auto spectrum = fft_timeseries_to_frequency(timeseries);
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
  fft_supercell_vector_field_to_kspace(data, kspace_data_, globals::lattice->size(), globals::lattice->kspace_size(), globals::lattice->num_motif_atoms());
  store_kspace_data_on_path(kspace_data_, kspace_paths_);
  periodogram_index_++;
}

void SpectrumBaseMonitor::print_info() const {
  std::cout << "\n";
  std::cout << "  number of samples "          << num_time_samples() << "\n";
  std::cout << "  sampling time (s) "          << sample_time_interval() << "\n";
  std::cout << "  acquisition time (s) "       << sample_time_interval() * num_time_samples() << "\n";
  std::cout << "  frequency resolution (THz) " << frequency_resolution_thz() << "\n";
  std::cout << "  maximum frequency (THz) "    << max_frequency_thz() << "\n";
  std::cout << "\n";
}