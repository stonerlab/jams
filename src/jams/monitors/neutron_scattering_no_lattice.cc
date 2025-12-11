//
// Created by Joseph Barker on 2018-11-22.
//

#include <jams/containers/vec3.h>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/cuda/cuda_common.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

#include "jams/monitors/spectrum_general.h"
#include "jams/monitors/neutron_scattering_no_lattice.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/neutrons.h"
#include <jams/helpers/mixed_precision.h>

NeutronScatteringNoLatticeMonitor::NeutronScatteringNoLatticeMonitor(const libconfig::Setting &settings)
: Monitor(settings),
 neartree_(globals::lattice->get_supercell().a1(),
           globals::lattice->get_supercell().a2(),
           globals::lattice->get_supercell().a3(), globals::lattice->periodic_boundaries(), globals::lattice->max_interaction_radius(), jams::defaults::lattice_tolerance)
{

  neartree_.insert_sites(globals::lattice->lattice_site_positions_cart());

  configure_kspace_vectors(settings);

  do_rspace_windowing_ = jams::config_optional(settings, "rspace_windowing", do_rspace_windowing_);
  std::cout << "rspace windowing: " << do_rspace_windowing_ << std::endl;

  // default to 1.0 in case no form factor is given in the settings
  fill(neutron_form_factors_.resize(globals::lattice->num_materials(), num_k_), 1.0);
  if (settings.exists("form_factor")) {
    configure_form_factors(settings["form_factor"]);
  }

  if (settings.exists("polarizations")) {
    configure_polarizations(settings["polarizations"]);
  }

  if (settings.exists("periodogram")) {
    configure_periodogram(settings["periodogram"]);
  }

  periodogram_props_.sample_time = output_step_freq_ * globals::solver->time_step();


  zero(spin_timeseries_.resize(globals::num_spins, 3, periodogram_props_.length));
  zero(spin_frequencies_.resize(globals::num_spins, 3, periodogram_props_.length));

  zero(kspace_spins_timeseries_.resize(periodogram_props_.length, kspace_path_.size()));
  zero(total_unpolarized_neutron_cross_section_.resize(
      periodogram_props_.length, kspace_path_.size()));
  zero(total_polarized_neutron_cross_sections_.resize(
      neutron_polarizations_.size(),periodogram_props_.length, kspace_path_.size()));
}

void NeutronScatteringNoLatticeMonitor::update(Solver& solver) {
  store_kspace_data_on_path();
  store_spin_data();
  periodogram_index_++;

  if (is_multiple_of(periodogram_index_, periodogram_props_.length)) {

//    auto spectrum = periodogram();

    shift_periodogram_overlap();
    total_periods_++;
//
//    element_sum(total_unpolarized_neutron_cross_section_, calculate_unpolarized_cross_section(spectrum));
//
//    if (!neutron_polarizations_.empty()) {
//      element_sum(total_polarized_neutron_cross_sections_,
//                  calculate_polarized_cross_sections(spectrum, neutron_polarizations_));
//    }


    output_fixed_spectrum();
//    output_neutron_cross_section();
//    output_static_structure_factor();
  }
}

void NeutronScatteringNoLatticeMonitor::configure_kspace_vectors(const libconfig::Setting &settings) {
  kmax_ = jams::config_required<double>(settings, "kmax");
  kvector_ = jams::config_required<Vec3>(settings, "kvector");
  num_k_ = jams::config_required<int>(settings, "num_k");

  kspace_path_.resize(num_k_ + 1);
  for (auto i = 0; i < kspace_path_.size(); ++i) {
    kspace_path_(i) = kvector_ * i * (kmax_ / num_k_);
  }

  rspace_displacement_.resize(globals::s.size(0));
  Vec3i lattice_dimensions = ::globals::lattice->size();
  for (auto i = 0; i < globals::s.size(0); ++i) {
    // generalize so that we can impose open boundaries
    rspace_displacement_(i) = globals::lattice->displacement(
        {0.5 * lattice_dimensions[0], 0.5 * lattice_dimensions[1], 0.5 * lattice_dimensions[2]},
        globals::lattice->lattice_site_position_cart(i));
  }
}

jams::MultiArray<jams::ComplexHi, 2>
NeutronScatteringNoLatticeMonitor::calculate_unpolarized_cross_section(const jams::MultiArray<Vec3cx,2> &spectrum) {
  const auto num_freqencies = spectrum.size(0);
  const auto num_reciprocal_points = kspace_path_.size();

  jams::MultiArray<jams::ComplexHi, 2> cross_section(num_freqencies, num_reciprocal_points);
  cross_section.zero();

  for (auto f = 0; f < num_freqencies; ++f) {
    for (auto k = 0; k < num_reciprocal_points; ++k) {
          auto Q = unit_vector(kspace_path_(k));
          auto s_a = conj(spectrum(f, k));
          auto s_b = spectrum(f, k);

          auto ff = pow2(neutron_form_factors_(0, k)); // NOTE: currently only supports one material
          for (auto i : {0, 1, 2}) {
            for (auto j : {0, 1, 2}) {
              cross_section(f, k) += ff * (kronecker_delta(i, j) - Q[i] * Q[j]) * (s_a[i] * s_b[j]);
            }
          }
        }
      }
  return cross_section;
}

jams::MultiArray<jams::ComplexHi, 3>
NeutronScatteringNoLatticeMonitor::calculate_polarized_cross_sections(const jams::MultiArray<Vec3cx, 2> &spectrum,
    const std::vector<Vec3> &polarizations) {
  const auto num_freqencies = spectrum.size(0);
  const auto num_reciprocal_points = kspace_path_.size();

  jams::MultiArray<jams::ComplexHi, 3> convolved(polarizations.size(), num_freqencies, num_reciprocal_points);
  convolved.zero();

  for (auto f = 0; f < num_freqencies; ++f) {
    for (auto k = 0; k < num_reciprocal_points; ++k) {
      auto Q = unit_vector(kspace_path_(k));
      auto s_a = conj(spectrum(f, k));
      auto s_b = spectrum(f, k);
      auto ff = pow2(neutron_form_factors_(0, k)); // NOTE: currently only supports one material

      for (auto p = 0; p < polarizations.size(); ++p) {
        auto P = polarizations[p];
        auto PxQ = cross(P, Q);

        convolved(p, f, k) += ff * kImagOne * dot(P, cross(s_a, s_b));

        for (auto i : {0, 1, 2}) {
          for (auto j : {0, 1, 2}) {
            convolved(p, f, k) += ff * kImagOne * PxQ[i] * Q[j] * (s_a[i] * s_b[j] - s_a[j] * s_b[i]);
          }
        }
      }
    }
  }
  return convolved;
}

jams::MultiArray<Vec3cx,2> NeutronScatteringNoLatticeMonitor::periodogram() {
  jams::MultiArray<Vec3cx,2> spectrum(kspace_spins_timeseries_);

  const int num_time_samples  = spectrum.size(0);
  const int num_kspace_samples = spectrum.size(1);

  int rank = 1;
  int transform_size[1] = {num_time_samples};
  int num_transforms = num_kspace_samples * 3;
  int nembed[1] = {num_time_samples};
  int stride = num_kspace_samples * 3;
  int dist = 1;

  fftw_plan fft_plan = fftw_plan_many_dft(rank, transform_size, num_transforms,
                                          FFTW_COMPLEX_CAST(spectrum.begin()), nembed, stride, dist,
                                          FFTW_COMPLEX_CAST(spectrum.begin()), nembed, stride, dist,
                                          FFTW_BACKWARD, FFTW_ESTIMATE);

  assert(fft_plan);

  jams::MultiArray<Vec3cx, 1> static_spectrum(num_kspace_samples);
  zero(static_spectrum);
  for (auto i = 0; i < num_time_samples; ++i) {
    for (auto j = 0; j < num_kspace_samples; ++j) {
      static_spectrum(j) += spectrum(i, j);
    }
  }
  element_scale(static_spectrum, 1.0/double(num_time_samples));


  for (auto i = 0; i < num_time_samples; ++i) {
    for (auto j = 0; j < num_kspace_samples; ++j) {
      spectrum(i, j) = fft_window_default(i, num_time_samples) *
          (spectrum(i, j) - static_spectrum(j));
    }
  }

  fftw_execute(fft_plan);
  fftw_destroy_plan(fft_plan);

  element_scale(spectrum, 1.0 / double(num_time_samples));

  return spectrum;
}

void NeutronScatteringNoLatticeMonitor::shift_periodogram_overlap() {
  // shift overlap data to the start of the range
  for (auto i = 0; i < periodogram_props_.overlap; ++i) {
    for (auto j = 0; j < kspace_spins_timeseries_.size(1); ++j) {
      kspace_spins_timeseries_(i, j) = kspace_spins_timeseries_(kspace_spins_timeseries_.size(0) - periodogram_props_.overlap + i, j);
    }
  }

  // put the pointer to the overlap position
  periodogram_index_ = periodogram_props_.overlap;
}

void NeutronScatteringNoLatticeMonitor::output_static_structure_factor() {
  auto static_structure_factor = calculate_static_structure_factor();
  const auto num_time_points = kspace_spins_timeseries_.size(0);


  std::ofstream ofs(jams::output::full_path_filename("static_structure_factor.tsv"));

  ofs << "index\t" << "qx\t" << "qy\t" << "qz\t" << "q_A-1\t";
  ofs << "Sxx_re\t" << "Sxx_im\t" << "Sxy_re\t" << "Sxy_im\t" << "Sxz_re\t" << "Sxz_im\t";
  ofs << "Syx_re\t" << "Syx_im\t" << "Syy_re\t" << "Syy_im\t" << "Syz_re\t" << "Syz_im\t";
  ofs << "Szx_re\t" << "Szx_im\t" << "Szy_re\t" << "Szy_im\t" << "Szz_re\t" << "Szz_im\n";

  for (auto k = 0; k < kspace_path_.size(); ++k) {
    ofs << jams::fmt::integer << k << "\t";
    ofs << jams::fmt::decimal << kspace_path_(k) << "\t";
    ofs << jams::fmt::decimal << kTwoPi * norm(kspace_path_(k)) / (
        globals::lattice->parameter() * 1e10) << "\t";
    for (auto i : {0,1,2}) {
      for (auto j : {0,1,2}) {
        auto s_a = static_structure_factor(k)[i] / double(num_time_points);
        auto s_b = static_structure_factor(k)[j] / double(num_time_points);
        auto s_ab = conj(s_a) * s_b;
        ofs << jams::fmt::sci << s_ab.real() << "\t";
        ofs << jams::fmt::sci << s_ab.imag() << "\t";
      }
    }
    ofs << "\n";
  }
  ofs.close();
}

void NeutronScatteringNoLatticeMonitor::output_neutron_cross_section() {
    std::ofstream ofs(jams::output::full_path_filename("neutron_scattering.tsv"));

    ofs << "index\t" << "qx\t" << "qy\t" << "qz\t" << "q_A-1\t";
    ofs << "freq_THz\t" << "energy_meV\t" << "sigma_unpol_re\t" << "sigma_unpol_im\t";
    for (auto k = 0; k < total_polarized_neutron_cross_sections_.size(0); ++k) {
      ofs << "sigma_pol" << std::to_string(k) << "_re\t" << "sigma_pol" << std::to_string(k) << "_im\t";
    }
    ofs << "\n";

    // sample time is here because the fourier transform in time is not an integral
    // but a discrete sum
    auto prefactor = (periodogram_props_.sample_time / double(total_periods_)) * (1.0 / (kTwoPi * kHBarIU))
                     * pow2((0.5 * kNeutronGFactor * pow2(kElementaryCharge)) / (kElectronMass * pow2(kSpeedOfLight)));
    auto barns_unitcell = prefactor / (1e-28);
    auto time_points = total_unpolarized_neutron_cross_section_.size(0);
    auto freq_delta = 1.0 / (periodogram_props_.length * periodogram_props_.sample_time);

    for (auto i = 0; i < (time_points / 2) + 1; ++i) {
      for (auto j = 0; j < kspace_path_.size(); ++j) {
        ofs << jams::fmt::integer << j << "\t";
        ofs << jams::fmt::decimal << kspace_path_(j) << "\t";
        ofs << jams::fmt::decimal << kTwoPi * norm(kspace_path_(j)) / (
            globals::lattice->parameter() * 1e10) << "\t";
        ofs << jams::fmt::decimal << (i * freq_delta) << "\t"; // THz
        ofs << jams::fmt::decimal << (i * freq_delta) * 4.135668 << "\t"; // meV
        // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
        ofs << jams::fmt::sci << barns_unitcell * total_unpolarized_neutron_cross_section_(i, j).real() << "\t";
        ofs << jams::fmt::sci << barns_unitcell * total_unpolarized_neutron_cross_section_(i, j).imag() << "\t";
        for (auto k = 0; k < total_polarized_neutron_cross_sections_.size(0); ++k) {
          ofs << jams::fmt::sci << barns_unitcell * total_polarized_neutron_cross_sections_(k, i, j).real() << "\t";
          ofs << jams::fmt::sci << barns_unitcell * total_polarized_neutron_cross_sections_(k, i, j).imag() << "\t";
        }
        ofs << "\n";
      }
      ofs << std::endl;
    }

    ofs.close();
}


void NeutronScatteringNoLatticeMonitor::store_kspace_data_on_path() {
  auto i = periodogram_index_;

  fill(&kspace_spins_timeseries_(i,0), &kspace_spins_timeseries_(i,0) + kspace_path_.size(), Vec3cx{0.0});

  for (auto n = 0; n < globals::num_spins; ++n) {
    Vec3 spin = {globals::s(n,0), globals::s(n,1), globals::s(n,2)};

    Vec3 r = rspace_displacement_(n);

    // this is effectively a window in rspace
    if (norm(r) >= globals::lattice->max_interaction_radius()) continue;
    auto delta_q = kspace_path_(1) - kspace_path_(0);

    auto f0 = exp(-kImagTwoPi * dot(delta_q, r));
    auto f = jams::ComplexHi{1.0, 0.0};
    for (auto k = 0; k < kspace_path_.size(); ++k) {
      kspace_spins_timeseries_(i, k) += f * spin;
      f *= f0;
    }
  }

  for (auto k = 0; k < kspace_path_.size(); ++k) {
    kspace_spins_timeseries_(i, k) /= double(globals::num_spins);
  }
}

void NeutronScatteringNoLatticeMonitor::configure_polarizations(libconfig::Setting &settings) {
  for (auto i = 0; i < settings.getLength(); ++i) {
    neutron_polarizations_.push_back({
                                         double{settings[i][0]}, double{settings[i][1]}, double{settings[i][2]}});
  }
}

void NeutronScatteringNoLatticeMonitor::configure_periodogram(libconfig::Setting &settings) {
  periodogram_props_.length = settings["length"];
  periodogram_props_.overlap = settings["overlap"];
}

void NeutronScatteringNoLatticeMonitor::configure_form_factors(libconfig::Setting &settings) {
  auto gj = jams::read_form_factor_settings(settings);

  auto num_materials = globals::lattice->num_materials();

  if (settings.getLength() != num_materials) {
    throw std::runtime_error("NeutronScatteringMonitor:: there must be one form factor per material\"");
  }

  std::vector<jams::FormFactorG> g_params(num_materials);
  std::vector<jams::FormFactorJ> j_params(num_materials);

  for (auto i = 0; i < settings.getLength(); ++i) {
    for (auto l : {0,2,4,6}) {
      j_params[i][l] = jams::config_optional<jams::FormFactorCoeff>(settings[i], "j" + std::to_string(l), j_params[i][l]);
    }
    g_params[i] = jams::config_required<jams::FormFactorG>(settings[i], "g");
  }

  neutron_form_factors_.resize(num_materials, num_k_);
  for (auto a = 0; a < num_materials; ++a) {
    for (auto i = 0; i < num_k_; ++i) {
      auto q = kspace_path_(i);
      neutron_form_factors_(a, i) = form_factor(q, kMeterToAngstroms * globals::lattice->parameter(), g_params[a], j_params[a]);
    }
  }
}

jams::MultiArray<Vec3cx,1> NeutronScatteringNoLatticeMonitor::calculate_static_structure_factor() {
  const auto num_time_points = kspace_spins_timeseries_.size(0);
  const auto num_k_points = kspace_spins_timeseries_.size(1);
  jams::MultiArray<Vec3cx,1> static_structure_factor(num_k_points);
  zero(static_structure_factor);
  for (auto i = 0; i < num_time_points; ++i) {
    for (auto j = 0; j < num_k_points; ++j) {
      static_structure_factor(j) += kspace_spins_timeseries_(i,j);
    }
  }
  return static_structure_factor;
}


void NeutronScatteringNoLatticeMonitor::store_spin_data() {
  auto t = periodogram_index_;

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      spin_timeseries_(i,j,t) = globals::s(i,j);
    }
  }

}

void NeutronScatteringNoLatticeMonitor::output_fixed_spectrum() {
  // Do temporal fourier transform of spin data

  const int num_time_samples = periodogram_props_.length;

  int rank = 1;
  int transform_size[1] = {num_time_samples};
  int num_transforms = globals::num_spins3;
  int nembed[1] = {num_time_samples};
  int stride = 1;
  int dist = num_time_samples;

  fftw_plan fft_plan = fftw_plan_many_dft(rank, transform_size, num_transforms,
                                          FFTW_COMPLEX_CAST(
                                              spin_frequencies_.data()),
                                          nembed, stride, dist,
                                          FFTW_COMPLEX_CAST(
                                              spin_frequencies_.data()),
                                          nembed, stride, dist,
                                          FFTW_BACKWARD, FFTW_ESTIMATE);

  assert(fft_plan);

  jams::MultiArray<double, 2> spin_averages(globals::num_spins, 3);
  zero(spin_averages);
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      for (auto t = 0; t < num_time_samples; ++t) {
        spin_averages(i, j) += spin_timeseries_(i, j, t);
      }
    }
  }
  element_scale(spin_averages, 1.0/double(num_time_samples));


  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      for (auto t = 0; t < num_time_samples; ++t) {
        spin_frequencies_(i, j, t) = fft_window_default(t, num_time_samples) * (spin_timeseries_(i, j, t) - spin_averages(i,j));
      }
    }
  }




  fftw_execute(fft_plan);

  std::ofstream debug(jams::output::full_path_filename("debug.tsv"));
  for (auto t = 0; t < num_time_samples; ++t) {
    debug << t << " " << spin_frequencies_(0, 0, t).real() << " " << spin_frequencies_(0, 0, t).imag() << " " << spin_frequencies_(0, 1, t).real() << " " << spin_frequencies_(0, 1, t).imag() << std::endl;
  }
  debug.close();


  element_scale(spin_frequencies_, 1.0 / double(num_time_samples));

  // Do S(Q,w) = sum_i sum_j exp(-iQ.R_i) exp(iQ.R_j) conj(S_i^a(w)) S_j^b(w)

  jams::MultiArray<std::complex<double>, 2> sqw(kspace_path_.size(),
                                                num_time_samples / 2 + 1);

  jams::MultiArray<std::complex<double>, 1> sw_i(num_time_samples / 2 + 1);
  jams::MultiArray<std::complex<double>, 1> sw_j(num_time_samples / 2 + 1);

  // this assumes out kspace_path is a single straight line
  const auto delta_q = kspace_path_(1) - kspace_path_(0);
  auto Q = unit_vector(delta_q);

  zero(sqw);
  for (auto i = 0; i < globals::num_spins; ++i) {
    std::cout << i << std::endl;

    // i == j is a special case because conj(S_i^a(w)) S_j^b(w) == conj(S_j^a(w)) S_i^b(w)
    zero(sw_i);
    for (auto m = 0; m < 3; ++m) {
      for (auto n = 0; n < 3; ++n) {
        const auto geom = (kronecker_delta(m, n) - Q[m] * Q[n]);
        if (geom == 0.0) continue;
        for (auto w = 0; w < num_time_samples / 2 + 1; ++w) {
          sw_i(w) += geom * conj(spin_frequencies_(i, m, w)) * spin_frequencies_(i, n, w);
        }
      }
    }

    for (auto k = 0; k < kspace_path_.size(); ++k) {
      for (auto w = 0; w < num_time_samples / 2 + 1; ++w) {
        sqw(k,w) += sw_i(w);
      }
    }

    // i > j
    for (auto j = i+1; j < globals::num_spins; ++j) {

      const Vec3 r_ij =  globals::lattice->displacement(i, j);

      zero(sw_i);
      zero(sw_j);
      for (auto m = 0; m < 3; ++m) {
        for (auto n = 0; n < 3; ++n) {
          const auto geom = (kronecker_delta(m, n) - Q[m] * Q[n]);
          if (geom == 0.0) continue;
          for (auto w = 0; w < num_time_samples / 2 + 1; ++w) {
            sw_i(w) += geom * conj(spin_frequencies_(i, m, w)) * spin_frequencies_(j, n, w);
          }
          for (auto w = 0; w < num_time_samples / 2 + 1; ++w) {
            sw_j(w) += geom * conj(spin_frequencies_(j, m, w)) * spin_frequencies_(i, n, w);
          }
        }
      }

      const auto f0 = exp(-kImagTwoPi * dot(delta_q, r_ij));
      auto f = jams::ComplexHi{1.0, 0.0};
      for (auto k = 0; k < kspace_path_.size(); ++k) {
        for (auto w = 0; w < num_time_samples / 2 + 1; ++w) {
          sqw(k,w) += (f * sw_i(w) + conj(f) * sw_j(w));
        }
        f *= f0;
      }
    }
  }
  std::ofstream ofs(jams::output::full_path_filename("neutron_scattering_fixed.tsv"));

  ofs << "index\t" << "qx\t" << "qy\t" << "qz\t" << "q_A-1\t";
  ofs << "freq_THz\t" << "energy_meV\t" << "sigma_unpol_re\t" << "sigma_unpol_im\t";
  ofs << "\n";

  // sample time is here because the fourier transform in time is not an integral
  // but a discrete sum
  auto prefactor = (periodogram_props_.sample_time / double(total_periods_)) * (1.0 / (kTwoPi * kHBarIU))
                   * pow2((0.5 * kNeutronGFactor * pow2(kElementaryCharge)) / (kElectronMass * pow2(kSpeedOfLight)));
  auto barns_unitcell = prefactor / (1e-28);
  auto freq_delta = 1.0 / (periodogram_props_.length * periodogram_props_.sample_time);

  for (auto w = 0; w <  num_time_samples / 2 + 1; ++w) {
    for (auto k = 0; k < kspace_path_.size(); ++k) {
      ofs << jams::fmt::integer << k << "\t";
      ofs << jams::fmt::decimal << kspace_path_(k) << "\t";
      ofs << jams::fmt::decimal << kTwoPi * norm(kspace_path_(k)) / (
          globals::lattice->parameter() * 1e10) << "\t";
      ofs << jams::fmt::decimal << (w * freq_delta) << "\t"; // THz
      ofs << jams::fmt::decimal << (w * freq_delta) * 4.135668 << "\t"; // meV
      // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
      ofs << jams::fmt::sci << barns_unitcell * sqw(k, w).real() << "\t";
      ofs << jams::fmt::sci << barns_unitcell * sqw(k, w).imag() << "\t";
      ofs << "\n";
    }
    ofs << std::endl;
  }

  ofs.close();

  fftw_destroy_plan(fft_plan);

}
