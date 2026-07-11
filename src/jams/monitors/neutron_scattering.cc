// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "jams/helpers/output.h"
#include "jams/core/solver.h"
#include "jams/helpers/error.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/interface/fft.h"
#include "jams/interface/config.h"
#include "neutron_scattering.h"
#include "jams/helpers/neutrons.h"
#include <jams/helpers/mixed_precision.h>

namespace {
constexpr const char* kNeutronCrossSectionUnits = "barn sr^-1 meV^-1 unitcell^-1";
}

NeutronScatteringMonitor::NeutronScatteringMonitor(const libconfig::Setting &settings)
: SpectrumBaseMonitor(settings) {

  enable_cuda_frequency_slices_backend_();
  validate_cuda_time_fft_backend_support_();

  // default to 1.0 in case no form factor is given in the settings
  fill(neutron_form_factors_.resize(globals::lattice->num_basis_sites(), num_k_points()), 1.0);
  if (settings.exists("form_factor")) {
    configure_form_factors(settings["form_factor"]);
  }

  if (settings.exists("polarizations")) {
    configure_polarizations(settings["polarizations"]);
  }

  zero(total_unpolarized_neutron_cross_section_.resize(
      num_frequencies(), num_k_points()));
  zero(total_polarized_neutron_cross_sections_.resize(
      neutron_polarizations_.size(), num_frequencies(), num_k_points()));

  print_info();
}

void NeutronScatteringMonitor::configure_form_factors(libconfig::Setting &settings) {
  auto gj = jams::read_form_factor_settings(settings);

  auto num_sites     = globals::lattice->num_basis_sites();
  neutron_form_factors_.resize(num_sites, num_k_points());
  for (auto a = 0; a < num_sites; ++a) {
    for (auto i = 0; i < num_k_points(); ++i) {
      auto m = globals::lattice->basis_site_atom(a).material_index;
      auto q = k_points_[i].xyz;
      neutron_form_factors_(a, i) = form_factor(q, kMeterToAngstroms * globals::lattice->parameter(), gj.first[m], gj.second[m]);
    }
  }
}

void NeutronScatteringMonitor::configure_polarizations(libconfig::Setting &settings) {
  for (auto i = 0; i < settings.getLength(); ++i) {
    neutron_polarizations_.push_back(jams::read_vec_setting<double, 3>(settings[i], "polarization"));
  }
}

void NeutronScatteringMonitor::update(Solver& solver) {
  const auto& spins = globals::s;
  store_sk_snapshot(spins);

  if (periodogram_window_complete()) {
    for (auto k = 0; k < num_k_points(); ++k) {
      accumulate_cross_sections_for_k(k, compute_frequency_spectrum_at_k(k));
    }

    output_neutron_cross_section();
    advance_periodogram_window();
  }
}

void NeutronScatteringMonitor::accumulate_cross_sections_for_k(
    const int k_index,
    const CmplxMappedSlice& spectrum) {
  const auto num_sites = spectrum.extent(0);
  if (spectrum.extent(2) < 3) {
    throw std::runtime_error("NeutronScatteringMonitor requires at least 3 channels");
  }

  const auto time_points = periodogram_length();
  if (spectrum.extent(1) < static_cast<std::size_t>(time_points)) {
    throw std::runtime_error("NeutronScatteringMonitor spectrum slice has too few frequency bins");
  }
  const auto frequency_count = num_frequencies();
  const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);
  auto kpoint = k_points_[k_index];
  auto Q = jams::unit_vector(kpoint.xyz);

  for (auto a = 0; a < num_sites; ++a) {
    for (auto b = 0; b < num_sites; ++b) {
      const auto ff = neutron_form_factors_(a, k_index) * neutron_form_factors_(b, k_index);
      const auto spin_scale = basis_spin_length_(a) * basis_spin_length_(b);
      const auto amplitude_scale = ff * spin_scale;

      for (auto freq = 0; freq < frequency_count; ++freq) {
        const auto f = keep_negative_frequencies() ? (freq_start + freq) % time_points : freq;
        // SpectrumBaseMonitor stores S_a(Q) with the full site-position phase,
        // so no additional motif-pair phase is applied here.
        jams::Vec<std::complex<double>, 3> s_a = {
            conj(spectrum(a, f, 0)),
            conj(spectrum(a, f, 1)),
            conj(spectrum(a, f, 2))
        };
        jams::Vec<std::complex<double>, 3> s_b = {
            spectrum(b, f, 0),
            spectrum(b, f, 1),
            spectrum(b, f, 2)
        };
        for (auto i : {0, 1, 2}) {
          for (auto j : {0, 1, 2}) {
            total_unpolarized_neutron_cross_section_(f, k_index) +=
                amplitude_scale * (kronecker_delta(i, j) - Q[i] * Q[j]) * s_a[i] * s_b[j];
          }
        }

        for (auto p = 0; p < neutron_polarizations_.size(); ++p) {
          auto P = neutron_polarizations_[p];
          auto PxQ = jams::cross(P, Q);

          total_polarized_neutron_cross_sections_(p, f, k_index) +=
              amplitude_scale * kImagOne * jams::dot(P, jams::cross(s_a, s_b));

          for (auto i : {0, 1, 2}) {
            for (auto j : {0, 1, 2}) {
              total_polarized_neutron_cross_sections_(p, f, k_index) +=
                  kImagOne * amplitude_scale * PxQ[i] * Q[j] * (s_a[i] * s_b[j] - s_a[j] * s_b[i]);
            }
          }
        }
      }
    }
  }
}

void NeutronScatteringMonitor::output_neutron_cross_section() {

  for (auto n = 0; n < k_segment_offsets_.size() - 1; ++n) {
    std::vector<jams::output::ColDef> cols = {
        {"index", "none", jams::output::ColFmt::Integer},
        {"q_total", "lattice constants^-1", jams::output::ColFmt::Fixed},
        {"h", "rlu", jams::output::ColFmt::Fixed},
        {"k", "rlu", jams::output::ColFmt::Fixed},
        {"l", "rlu", jams::output::ColFmt::Fixed},
        {"qx", "lattice constants^-1", jams::output::ColFmt::Fixed},
        {"qy", "lattice constants^-1", jams::output::ColFmt::Fixed},
        {"qz", "lattice constants^-1", jams::output::ColFmt::Fixed},
        {"freq_THz", "THz", jams::output::ColFmt::Fixed},
        {"energy_meV", "meV", jams::output::ColFmt::Fixed},
        {"sigma_unpol_re", kNeutronCrossSectionUnits},
        {"sigma_unpol_im", kNeutronCrossSectionUnits}};
    for (auto k = 0; k < total_polarized_neutron_cross_sections_.extent(0); ++k) {
      cols.push_back({"sigma_pol" + std::to_string(k) + "_re", kNeutronCrossSectionUnits});
      cols.push_back({"sigma_pol" + std::to_string(k) + "_im", kNeutronCrossSectionUnits});
    }
    jams::output::TsvWriter tsv(
        jams::output::monitor_filename_series(name() + "_path", "tsv", n),
        std::move(cols));

    const auto barns_unitcell = jams::neutron_cross_section_barn_mev_scale(
        sample_time_interval(),
        periodogram_length(),
        periodogram_window_count(),
        static_cast<int>(globals::lattice->num_cells()));
    const auto time_points = periodogram_length();
    const auto freq_count = total_unpolarized_neutron_cross_section_.extent(0);
    const auto freq_start = (time_points % 2 == 0) ? (time_points / 2 + 1) : ((time_points + 1) / 2);

    auto path_begin = k_segment_offsets_[n];
    auto path_end = k_segment_offsets_[n + 1];
    for (auto i = 0; i < freq_count; ++i) {
      const auto f = keep_negative_frequencies() ? (freq_start + i) % time_points : i;
      const auto freq_index = (f <= time_points / 2) ? static_cast<int>(f)
                                                     : static_cast<int>(f) - static_cast<int>(time_points);
      const auto freq_thz = static_cast<double>(freq_index) * frequency_resolution_thz();
      double total_distance = 0.0;
      for (auto j = path_begin; j < path_end; ++j) {
        std::vector<double> values;
        values.reserve(tsv.num_cols());
        values.push_back(j);
        values.push_back(total_distance);
        values.push_back(k_points_[j].hkl[0]);
        values.push_back(k_points_[j].hkl[1]);
        values.push_back(k_points_[j].hkl[2]);
        values.push_back(k_points_[j].xyz[0]);
        values.push_back(k_points_[j].xyz[1]);
        values.push_back(k_points_[j].xyz[2]);
        values.push_back(freq_thz);
        values.push_back(freq_thz * kTHz2meV);
        values.push_back(barns_unitcell * total_unpolarized_neutron_cross_section_(f, j).real());
        values.push_back(barns_unitcell * total_unpolarized_neutron_cross_section_(f, j).imag());
        for (auto k = 0; k < total_polarized_neutron_cross_sections_.extent(0); ++k) {
          values.push_back(barns_unitcell * total_polarized_neutron_cross_sections_(k, f, j).real());
          values.push_back(barns_unitcell * total_polarized_neutron_cross_sections_(k, f, j).imag());
        }
        tsv.write_row(values);
        if (j + 1 < path_end) {
          total_distance += jams::norm(k_points_[j].xyz - k_points_[j + 1].xyz);
        }
      }
    }
  }
}
