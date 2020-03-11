//
// Created by Joseph Barker on 2019-08-01.
//

#include <jams/helpers/output.h>
#include "jams/core/lattice.h"
#include "jams/core/globals.h"
#include "jams/monitors/magnon_spectrum.h"

using namespace std;
using namespace jams;
using namespace libconfig;

MagnonSpectrumMonitor::MagnonSpectrumMonitor(const libconfig::Setting &settings) : SpectrumBaseMonitor(settings) {
  zero(cumulative_magnon_spectrum_.resize(num_motif_atoms(), num_time_samples(), num_kpoints()));
  zero(transformations_.resize(globals::num_spins));
  zero(transformed_spins_.resize(globals::num_spins, 3));

  for (auto i = 0; i < globals::num_spins; ++i) {
    transformations_(i) = lattice->material(lattice->atom_material_id(i)).transform;
  }

  print_info();
}

void MagnonSpectrumMonitor::update(Solver *solver) {

  transformed_spins_.zero();

  for (auto n = 0; n < globals::num_spins; ++n) {
    for (auto i = 0; i < 3; ++i) {
      for (auto j = 0; j < 3; ++j) {
        transformed_spins_(n, i) += transformations_(n)[i][j] * globals::s(n, j);
      }
    }
  }

  store_periodogram_data(transformed_spins_);

  if (do_periodogram_update()) {
    auto spectrum = compute_periodogram_spectrum(kspace_data_timeseries_);

    element_sum(cumulative_magnon_spectrum_,
                calculate_magnon_spectrum(spectrum));

    output_magnon_spectrum();
  }
}

void MagnonSpectrumMonitor::output_magnon_spectrum() {
  for (auto n = 0; n < kspace_continuous_path_ranges_.size() - 1; ++n) {
    ofstream ofs = jams::output::open_file(simulation_name + "_magnon_spectrum_path_" + to_string(n) + ".tsv");

    ofs << "index\t" << "h\t" << "k\t" << "l\t" << "qx\t" << "qy\t" << "qz\t";
    ofs << "freq_THz\t" << "energy_meV\t";
    ofs << "sqw_x_re\t" << "sqw_x_re_im\t";
    ofs << "sqw_y_re\t" << "sqw_y_re_im\t";
    ofs << "sqw_z_re\t" << "sqw_z_re_im\t";
    ofs << "\n";

    // sample time is here because the fourier transform in time is not an integral
    // but a discrete sum
    auto prefactor = (sample_time_interval() / num_periodogram_iterations());
    auto time_points = cumulative_magnon_spectrum_.size(1);

    MultiArray<MagnonSpectrumMonitor::Mat3cx, 2> total_magnon_spectrum(cumulative_magnon_spectrum_.size(1), cumulative_magnon_spectrum_.size(2));
    zero(total_magnon_spectrum);
    for (auto a = 0; a < cumulative_magnon_spectrum_.size(0); ++a) {
      for (auto f = 0; f < cumulative_magnon_spectrum_.size(1); ++f) {
        for (auto k = 0; k < cumulative_magnon_spectrum_.size(2); ++k) {
          total_magnon_spectrum(f, k) += cumulative_magnon_spectrum_(a, f, k);
        }
      }
    }

    auto path_begin = kspace_continuous_path_ranges_[n];
    auto path_end = kspace_continuous_path_ranges_[n + 1];
    for (auto i = 0; i < (time_points / 2) + 1; ++i) {
      for (auto j = path_begin; j < path_end; ++j) {
        ofs << fmt::integer << j << "\t";
        ofs << fmt::decimal << kspace_paths_[j].hkl << "\t";
        ofs << fmt::decimal << kspace_paths_[j].xyz << "\t";
        ofs << fmt::decimal << i * frequency_resolution_thz() << "\t"; // THz
        ofs << fmt::decimal << i * frequency_resolution_thz() * 4.135668 << "\t"; // meV
        // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
        for (auto k : {0,1,2}) {
          for (auto l : {0,1,2}) {
            ofs << fmt::sci << prefactor * total_magnon_spectrum(i, j)[k][l].real() << "\t";
            ofs << fmt::sci << prefactor * total_magnon_spectrum(i, j)[k][l].imag() << "\t";
          }
        }
        ofs << "\n";
      }
      ofs << endl;
    }

    ofs.close();
  }
}

jams::MultiArray<MagnonSpectrumMonitor::Mat3cx, 3>
MagnonSpectrumMonitor::calculate_magnon_spectrum(const jams::MultiArray<Vec3cx, 3> &spectrum) {
  const auto num_sites = spectrum.size(0);
  const auto num_freqencies = spectrum.size(1);
  const auto num_reciprocal_points = spectrum.size(2);

  MultiArray<Mat3cx, 3> magnon_spectrum(num_sites, num_freqencies, num_reciprocal_points);
  magnon_spectrum.zero();

  for (auto a = 0; a < num_sites; ++a) {
    // structure factor: note that q and r are in fractional coordinates (hkl, abc)
    const Vec3 r = lattice->motif_atom(a).position;
      for (auto k = 0; k < num_reciprocal_points; ++k) {
        auto kpoint = kspace_paths_[k];
        auto q = kpoint.hkl;
        for (auto f = 0; f < num_freqencies; ++f) {
          auto sqw = spectrum(a, f, k) * exp(-kImagTwoPi * dot(q, r));
          for (auto i : {0, 1, 2}) {
            for (auto j : {0, 1, 2}) {
              magnon_spectrum(a, f, k)[i][j] = conj(sqw[i]) * sqw[j];
            }
          }
        }
    }
  }
  return magnon_spectrum;
}

