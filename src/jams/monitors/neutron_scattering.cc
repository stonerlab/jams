// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "jams/core/solver.h"
#include "jams/helpers/error.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/interface/fft.h"
#include "neutron_scattering.h"

using namespace std;
using namespace jams;
using libconfig::Setting;
using Complex = std::complex<double>;

namespace jams {
    struct FormFactorCoeff { double A, a, B, b, C, c, D; };
    struct FormFactorG { double g0, g2, g4, g6; };
    struct FormFactorJ { FormFactorCoeff j0, j2, j4, j6; };

    template<>
    inline FormFactorCoeff config_required(const Setting &s, const string &name) {
      return FormFactorCoeff{double{s[name][0]}, double{s[name][1]}, double{s[name][2]}, double{s[name][3]},
                             double{s[name][4]}, double{s[name][5]}, double{s[name][6]}};
    }

    template<>
    inline FormFactorG config_required(const Setting &s, const string &name) {
      return FormFactorG{double{s[name][0]}, double{s[name][1]}, double{s[name][2]}, double{s[name][3]}};
    }

    // Calculates the approximate neutron form factor at |q|
    double form_factor(const Vec3 &q, const FormFactorG &g, const FormFactorJ &j) {
      auto a = kMeterToAngstroms * lattice->parameter();
      auto s = norm(q) / (4.0 * kPi * a);

      // Approximation and constants from International Tables for Crystallography: Vol. C (pp. 454–461).
      auto jls = [](const int &l, const double &s, const FormFactorCoeff &f) {
          double p = (l == 0) ? 1.0 : s * s;
          return p * (f.A * exp(-f.a * s * s) + f.B * exp(-f.b * s * s) + f.C * exp(-f.c * s * s) + f.D);
      };

      return 0.5 * (g.g0 * jls(0, s, j.j0) + g.g2 * jls(2, s, j.j2)
                  + g.g4 * jls(4, s, j.j4) + g.g6 * jls(6, s, j.j6));
    }
}

NeutronScatteringMonitor::NeutronScatteringMonitor(const Setting &settings)
: Monitor(settings) {

  periodogram_props_.sample_time = output_step_freq_ * solver->time_step();
  t_sample_ = output_step_freq_ * solver->time_step();
  num_t_samples_ = periodogram_props_.length;
  freq_delta_ = 1.0 / (num_t_samples_ * periodogram_props_.sample_time);

  auto kspace_size   = lattice->kspace_size();
  auto num_sites     = lattice->num_motif_atoms();

  configure_kspace_paths(settings["hkl_path"]);
  configure_form_factors(settings["form_factor"]);

  neutron_polarizations_ = {Vec3{0, 0, 1}, Vec3{0, 0, -1}};

  zero(kspace_spins_.resize(kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_sites));
  zero(kspace_spins_timeseries_.resize(num_sites, periodogram_props_.length, kspace_paths_.size()));
  zero(total_polarized_neutron_cross_sections_.resize(neutron_polarizations_.size(),periodogram_props_.length, kspace_paths_.size()));
  zero(total_unpolarized_neutron_cross_section_.resize(periodogram_props_.length, kspace_paths_.size()));

  print_info();
}

void NeutronScatteringMonitor::update(Solver * solver) {
  const auto num_sites = ::lattice->num_motif_atoms();

  fft_supercell_vector_field_to_kspace(globals::s, kspace_spins_, lattice->kspace_size(), num_sites);
  store_kspace_data_on_path();
  periodogram_index_++;

  if (is_multiple_of(periodogram_index_, periodogram_props_.length)) {
    // calculate periodogram
    auto spectrum = periodogram();
    shift_periodogram_overlap();
    total_periods_++;

    element_sum(total_unpolarized_neutron_cross_section_,
                calculate_unpolarized_cross_section(spectrum));

    if (!neutron_polarizations_.empty()) {
      element_sum(total_polarized_neutron_cross_sections_,
                  calculate_polarized_cross_sections(spectrum, neutron_polarizations_));
    }

    output_neutron_cross_section();
  }
}

MultiArray<Vec3cx,3> NeutronScatteringMonitor::periodogram() {
  MultiArray<Vec3cx,3> spectrum(kspace_spins_timeseries_);

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
    fftw_plan fft_plan = fftw_plan_many_dft(
        rank, transform_size, num_transforms,
        FFTW_COMPLEX_CAST(&spectrum(a,0,0)), nembed, stride, dist,
        FFTW_COMPLEX_CAST(&spectrum(a,0,0)), nembed, stride, dist,
        FFTW_BACKWARD, FFTW_ESTIMATE);

    assert(fft_plan);

    for (auto i = 0; i < num_time_samples; ++i) {
      for (auto j = 0; j < num_space_samples; ++j) {
        spectrum(a, i, j) *= fft_window_default(i, num_time_samples);
      }
    }

    fftw_execute(fft_plan);
    fftw_destroy_plan(fft_plan);
  }

  element_scale(spectrum, 1.0 / double(num_time_samples));

  return spectrum;
}

void NeutronScatteringMonitor::output_neutron_cross_section() {
  for (auto n = 0; n < kspace_continuous_path_ranges_.size() - 1; ++n) {
    std::ofstream ofs(seedname + "_neutron_scattering_path_" + to_string(n) + ".tsv");

    ofs << "index\t" << "h\t" << "k\t" << "l\t" << "qx\t" << "qy\t" << "qz\t";
    ofs << "freq_THz\t" << "energy_meV\t" << "sigma_unpol_re\t" << "sigma_unpol_im\t";
    for (auto k = 0; k < total_polarized_neutron_cross_sections_.size(0); ++k) {
      ofs << "sigma_pol" << to_string(k) << "_re\t";
      ofs << "sigma_pol" << to_string(k) << "_im\t";
    }
    ofs << "\n";

    auto format_int = [](ostream &os) -> ostream & { return os << fixed; };
    auto format_fix = [](ostream &os) -> ostream & { return os << setprecision(8) << setw(12) << fixed;};
    auto format_sci = [](ostream &os) -> ostream & { return os << setprecision(8) << setw(12) << scientific;};

    // sample time is here because the fourier transform in time is not an integral
    // but a discrete sum
    double prefactor = t_sample_ * (1.0 / (kTwoPi * kHBar))
                       * pow2((0.5 * kNeutronGFactor * pow2(kElementaryCharge)) / (kElectronMass * pow2(kSpeedOfLight)));

    double barns_unitcell = prefactor / (1e-28 * lattice->num_cells());
    const auto time_points = total_unpolarized_neutron_cross_section_.size(0);

    auto path_begin = kspace_continuous_path_ranges_[n];
    auto path_end = kspace_continuous_path_ranges_[n + 1];

    // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
    for (auto i = 0; i < (time_points / 2) + 1; ++i) {
      for (auto j = path_begin; j < path_end; ++j) {
        ofs << format_int << j << "\t";
        ofs << format_fix << kspace_paths_[j].hkl << "\t";
        ofs << format_fix << kspace_paths_[j].xyz << "\t";
        ofs << format_fix << (i * freq_delta_ / 1e12) << "\t"; // THz
        ofs << format_fix << (i * freq_delta_ / 1e12) * 4.135668 << "\t"; // meV
        ofs << format_sci << barns_unitcell * total_unpolarized_neutron_cross_section_(i, j).real() / double(total_periods_) << "\t";
        ofs << format_sci << barns_unitcell * total_unpolarized_neutron_cross_section_(i, j).imag() / double(total_periods_) << "\t";
        for (auto k = 0; k < total_polarized_neutron_cross_sections_.size(0); ++k) {
          ofs << format_sci << barns_unitcell * total_polarized_neutron_cross_sections_(k, i, j).real() / double(total_periods_) << "\t";
          ofs << format_sci << barns_unitcell * total_polarized_neutron_cross_sections_(k, i, j).imag() / double(total_periods_) << "\t";
        }
        ofs << "\n";
      }
      ofs << endl;
    }

    ofs.close();
  }
}

void NeutronScatteringMonitor::configure_kspace_paths(Setting& settings) {
  // hkl_path can be a simple list of nodes e.g.
  //     hkl_path = ( [3.0, 3.0,-3.0], [ 5.0, 5.0,-5.0] );
  // or a list of discontinuous paths e.g.
  //    hkl_path = ( ([3.0, 3.0,-3.0], [ 5.0, 5.0,-5.0]),
  //                 ([3.0, 3.0,-2.0], [ 5.0, 5.0,-4.0]));

  if (!(settings[0].isList() || settings[0].isArray())) {
    jams_die("hkl_nodes in neutron-scattering monitor must be a list or a group");
  }

  bool has_discontinuous_paths = settings[0].isList();

  kspace_continuous_path_ranges_.push_back(0);

  if (has_discontinuous_paths) {
    for (auto n = 0; n < settings.getLength(); ++n) {
      vector<Vec3> hkl_path_nodes(settings[n].getLength());
      for (auto i = 0; i < settings[n].getLength(); ++i) {
        hkl_path_nodes[i] = Vec3{settings[n][i][0], settings[n][i][1], settings[n][i][2]};
      }

      auto new_path = generate_hkl_kspace_path(hkl_path_nodes, lattice->kspace_size());
      kspace_paths_.insert(end(kspace_paths_), begin(new_path), end(new_path));
      kspace_continuous_path_ranges_.push_back(kspace_continuous_path_ranges_.back() + new_path.size());
    }
  } else {
    vector<Vec3> hkl_path_nodes(settings.getLength());
    for (auto i = 0; i < settings.getLength(); ++i) {
      hkl_path_nodes[i] = Vec3{settings[i][0], settings[i][1], settings[i][2]};
    }

    kspace_paths_ = generate_hkl_kspace_path(hkl_path_nodes, lattice->kspace_size());
    kspace_continuous_path_ranges_.push_back(kspace_continuous_path_ranges_.back() + kspace_paths_.size());
  }
}

void NeutronScatteringMonitor::configure_form_factors(Setting &settings) {
  auto num_sites     = lattice->num_motif_atoms();
  auto num_materials = lattice->num_materials();

  if (settings.getLength() != num_materials) {
    jams_die("In NeutronScatteringMonitor there must be one form factor per material");
  }

  vector<FormFactorG> g_params(num_materials);
  vector<FormFactorJ> j_params(num_materials);

  for (auto i = 0; i < settings.getLength(); ++i) {
    j_params[i].j0 = config_optional<FormFactorCoeff>(settings[i], "j0", j_params[i].j0);
    j_params[i].j2 = config_optional<FormFactorCoeff>(settings[i], "j1", j_params[i].j2);
    j_params[i].j4 = config_optional<FormFactorCoeff>(settings[i], "j2", j_params[i].j4);
    j_params[i].j6 = config_optional<FormFactorCoeff>(settings[i], "j3", j_params[i].j6);
    g_params[i] = config_required<FormFactorG>(settings[i], "g");
  }

  neutron_form_factors_.resize(num_sites, kspace_paths_.size());
  for (auto a = 0; a < num_sites; ++a) {
    for (auto i = 0; i < kspace_paths_.size(); ++i) {
      auto m = lattice->motif_atom(a).material;
      auto q = kspace_paths_[i].xyz;
      neutron_form_factors_(a, i) = form_factor(q, g_params[m], j_params[m]);
    }
  }
}

void NeutronScatteringMonitor::print_info() {
  cout << "\n";
  cout << "  number of samples "          << periodogram_props_.length << "\n";
  cout << "  sampling time (s) "          << output_step_freq_ * solver->time_step() << "\n";
  cout << "  acquisition time (s) "       << periodogram_props_.sample_time * periodogram_props_.length << "\n";
  cout << "  frequency resolution (THz) " << (1.0 / (num_t_samples_ * periodogram_props_.sample_time)) / kTHz << "\n";
  cout << "  maximum frequency (THz) "    << (1.0 / (2.0 * periodogram_props_.sample_time)) / kTHz << "\n";
  cout << "\n";
}


/**
 * Generate a path between nodes in reciprocal space sampling the kspace discretely.
 *
 * @param hkl_nodes
 * @param kspace_size
 * @return
 */
vector<HKLIndex> NeutronScatteringMonitor::generate_hkl_kspace_path(const vector<Vec3> &hkl_nodes, const Vec3i &kspace_size) {
  vector<HKLIndex> hkl_path;
  for (auto n = 0; n < hkl_nodes.size()-1; ++n) {
    Vec3i origin = to_int(scale(hkl_nodes[n], kspace_size));
    Vec3i displacement = to_int(scale(hkl_nodes[n+1], kspace_size)) - origin;
    Vec3i delta = normalize_components(displacement);

    // use +1 to include the last point on the displacement
    const auto num_coordinates = abs_max(displacement) + 1;

    Vec3i coordinate = origin;
    for (auto i = 0; i < num_coordinates; ++i) {
      // map an arbitrary coordinate into the limited k indicies of the reduced brillouin zone
      Vec3 hkl = scale(coordinate, 1.0/to_double(kspace_size));
      Vec3 xyz = lattice->get_unitcell().inverse_matrix() * hkl;
      hkl_path.push_back(HKLIndex{hkl, xyz, fftw_r2c_index(coordinate, kspace_size)});

      coordinate += delta;
    }
  }
  // remove duplicates in the path where start and end indicies are the same at nodes
  hkl_path.erase(unique(hkl_path.begin(), hkl_path.end()), hkl_path.end());

  return hkl_path;
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
      Vec3 r_ab = lattice->motif_atom(b).fractional_pos - lattice->motif_atom(a).fractional_pos;

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
      const Vec3 r_ab = lattice->motif_atom(b).fractional_pos - lattice->motif_atom(a).fractional_pos;
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

void NeutronScatteringMonitor::shift_periodogram_overlap() {
  // shift overlap data to the start of the range
  for (auto a = 0; a < kspace_spins_timeseries_.size(0); ++a) {
    for (auto i = 0; i < periodogram_props_.overlap; ++i) {
      for (auto j = 0; j < kspace_spins_timeseries_.size(2); ++j) {
        kspace_spins_timeseries_(a, i, j) = kspace_spins_timeseries_(a, kspace_spins_timeseries_.size(1) - periodogram_props_.overlap + i, j);
      }
    }
  }
  // put the pointer to the overlap position
  periodogram_index_ = periodogram_props_.overlap;
}

void NeutronScatteringMonitor::store_kspace_data_on_path() {
  for (auto a = 0; a < kspace_spins_.size(3); ++a) {
    for (auto k = 0; k < kspace_paths_.size(); ++k) {
      auto kindex = kspace_paths_[k].index;
      auto i = periodogram_index_;
      auto idx = kindex.offset;
      if (kindex.conj) {
        kspace_spins_timeseries_(a, i, k) = conj(kspace_spins_(idx[0], idx[1], idx[2], a));
      } else {
        kspace_spins_timeseries_(a, i, k) = kspace_spins_(idx[0], idx[1], idx[2], a);
      }
    }
  }
}
