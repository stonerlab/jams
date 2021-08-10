// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "jams/helpers/output.h"
#include "jams/core/solver.h"
#include "jams/helpers/error.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/interface/fft.h"
#include "neutron_scattering.h"
#include "jams/helpers/neutrons.h"

using namespace std;
using namespace jams;
using libconfig::Setting;
using Complex = std::complex<double>;

NeutronScatteringMonitor::NeutronScatteringMonitor(const Setting &settings)
: SpectrumBaseMonitor(settings) {

  // default to 1.0 in case no form factor is given in the settings
  fill(neutron_form_factors_.resize(lattice->num_motif_atoms(), num_kpoints()), 1.0);
  if (settings.exists("form_factor")) {
    configure_form_factors(settings["form_factor"]);
  }

  if (settings.exists("polarizations")) {
    configure_polarizations(settings["polarizations"]);
  }

  zero(total_unpolarized_neutron_cross_section_.resize(
      num_time_samples(), num_kpoints()));
  zero(total_polarized_neutron_cross_sections_.resize(
      neutron_polarizations_.size(), num_time_samples(), num_kpoints()));

  print_info();
}

void NeutronScatteringMonitor::configure_form_factors(Setting &settings) {
  auto gj = read_form_factor_settings(settings);

  auto num_sites     = lattice->num_motif_atoms();
  neutron_form_factors_.resize(num_sites, num_kpoints());
  for (auto a = 0; a < num_sites; ++a) {
    for (auto i = 0; i < num_kpoints(); ++i) {
      auto m = lattice->motif_atom(a).material_index;
      auto q = kspace_paths_[i].xyz;
      neutron_form_factors_(a, i) = form_factor(q, kMeterToAngstroms * lattice->parameter(), gj.first[m], gj.second[m]);
    }
  }
}

void NeutronScatteringMonitor::configure_polarizations(libconfig::Setting &settings) {
  for (auto i = 0; i < settings.getLength(); ++i) {
    neutron_polarizations_.push_back({
                                         double{settings[i][0]}, double{settings[i][1]}, double{settings[i][2]}});
  }
}

void NeutronScatteringMonitor::update(Solver * solver) {
  store_periodogram_data(globals::s);

  if (do_periodogram_update()) {
    auto spectrum = compute_periodogram_spectrum(kspace_data_timeseries_);

    element_sum(total_unpolarized_neutron_cross_section_,
        calculate_unpolarized_cross_section(spectrum));

    if (!neutron_polarizations_.empty()) {
      element_sum(total_polarized_neutron_cross_sections_,
                  calculate_polarized_cross_sections(spectrum, neutron_polarizations_));
    }

    output_neutron_cross_section();
  }
}

/**
 * Compute the alpha, beta components of the scattiner cross section.
 *
 * @param alpha cartesian component {x,y,z}
 * @param beta  cartesian component {x,y,z}
 * @param site_a unit cell site index
 * @param site_b unit cell site index
 * @param sqw_a spin data for site a in reciprocal space and frequency space
 * @param sqw_b spin data for site b in reciprocal space and frequency space
 * @param hkl_indicies list of reciprocal space hkl indicies
 * @return
 */

MultiArray<Complex, 2> NeutronScatteringMonitor::calculate_unpolarized_cross_section(const MultiArray<Vec3cx, 3>& spectrum) {
  const auto num_sites = spectrum.size(0);
  const auto num_freqencies = spectrum.size(1);
  const auto num_reciprocal_points = spectrum.size(2);

  MultiArray<Complex, 2> cross_section(num_freqencies, num_reciprocal_points);
  cross_section.zero();

  for (auto a = 0; a < num_sites; ++a) {
    for (auto b = 0; b < num_sites; ++b) {
      Vec3 r_ab = lattice->motif_atom(b).position - lattice->motif_atom(a).position;

      for (auto k = 0; k < num_reciprocal_points; ++k) {
        auto kpoint = kspace_paths_[k];
        auto Q = unit_vector(kpoint.xyz);
        auto q = kpoint.hkl;
        auto ff = neutron_form_factors_(a, k) * neutron_form_factors_(b, k);
        // structure factor: note that q and r are in fractional coordinates (hkl, abc)
        auto sf = exp(-kImagTwoPi * dot(q, r_ab));

        for (auto f = 0; f < num_freqencies; ++f) {
          auto s_a = conj(spectrum(a, f, k));
          auto s_b = spectrum(b, f, k);
          for (auto i : {0, 1, 2}) {
            for (auto j : {0, 1, 2}) {
              cross_section(f, k) += sf * ff * (kronecker_delta(i, j) - Q[i] * Q[j]) * s_a[i] * s_b[j];
            }
          }
        }
      }
    }
  }
  return cross_section;
}

MultiArray<Complex, 3> NeutronScatteringMonitor::calculate_polarized_cross_sections(const MultiArray<Vec3cx, 3>& spectrum, const vector<Vec3>& polarizations) {
  const auto num_sites = spectrum.size(0);
  const auto num_freqencies = spectrum.size(1);
  const auto num_reciprocal_points = spectrum.size(2);

  MultiArray<Complex, 3> convolved(polarizations.size(), num_freqencies, num_reciprocal_points);
  convolved.zero();

  for (auto a = 0; a < num_sites; ++a) {
    for (auto b = 0; b < num_sites; ++b) {
      const Vec3 r_ab = lattice->motif_atom(b).position - lattice->motif_atom(a).position;
      for (auto k = 0; k < num_reciprocal_points; ++k) {
        auto kpoint = kspace_paths_[k];
        auto Q = unit_vector(kpoint.xyz);
        auto q = kpoint.hkl;
        auto ff = neutron_form_factors_(a, k) * neutron_form_factors_(b, k);
        // structure factor: note that q and r are in fractional coordinates (hkl, abc)
        auto sf = exp(-kImagTwoPi * dot(q, r_ab));

        for (auto f = 0; f < num_freqencies; ++f) {
          auto s_a = conj(spectrum(a, f, k));
          auto s_b = spectrum(b, f, k);
          for (auto p = 0; p < polarizations.size(); ++p) {
            auto P = polarizations[p];
            auto PxQ = cross(P, Q);

            convolved(p, f, k) += sf * ff * kImagOne * dot(P, cross(s_a, s_b));

            for (auto i : {0, 1, 2}) {
              for (auto j : {0, 1, 2}) {
                convolved(p, f, k) += kImagOne * sf * ff * PxQ[i] * Q[j] * ( s_a[i] * s_b[j] - s_a[j] * s_b[i]);
              }
            }
          }
        }
      }
    }
  }

  return convolved;
}

void NeutronScatteringMonitor::output_neutron_cross_section() {
  for (auto n = 0; n < kspace_continuous_path_ranges_.size() - 1; ++n) {
    std::ofstream ofs(jams::output::full_path_filename_series("neutron_scattering_path.tsv", n));

    ofs << "index\t" << "h\t" << "k\t" << "l\t" << "qx\t" << "qy\t" << "qz\t";
    ofs << "freq_THz\t" << "energy_meV\t" << "sigma_unpol_re\t" << "sigma_unpol_im\t";
    for (auto k = 0; k < total_polarized_neutron_cross_sections_.size(0); ++k) {
      ofs << "sigma_pol" << to_string(k) << "_re\t" << "sigma_pol" << to_string(k) << "_im\t";
    }
    ofs << "\n";

    // sample time is here because the fourier transform in time is not an integral
    // but a discrete sum
    auto prefactor = (sample_time_interval() / num_periodogram_iterations()) * (1.0 / (kTwoPi * kHBarIU))
                     * pow2((0.5 * kNeutronGFactor * pow2(kElementaryCharge)) / (kElectronMass * pow2(kSpeedOfLight)));
    auto barns_unitcell = prefactor / (1e-28 * lattice->num_cells());
    auto time_points = total_unpolarized_neutron_cross_section_.size(0);

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
        ofs << fmt::sci << barns_unitcell * total_unpolarized_neutron_cross_section_(i, j).real() << "\t";
        ofs << fmt::sci << barns_unitcell * total_unpolarized_neutron_cross_section_(i, j).imag() << "\t";
        for (auto k = 0; k < total_polarized_neutron_cross_sections_.size(0); ++k) {
          ofs << fmt::sci << barns_unitcell * total_polarized_neutron_cross_sections_(k, i, j).real() << "\t";
          ofs << fmt::sci << barns_unitcell * total_polarized_neutron_cross_sections_(k, i, j).imag() << "\t";
        }
        ofs << "\n";
      }
      ofs << endl;
    }

    ofs.close();
  }
}
