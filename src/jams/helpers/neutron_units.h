#ifndef JAMS_HELPERS_NEUTRON_UNITS_H
#define JAMS_HELPERS_NEUTRON_UNITS_H

#include <cmath>
#include <map>
#include <stdexcept>

#include "jams/containers/vec3.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"

namespace jams {
    struct FormFactorCoeff { double A, a, B, b, C, c, D; };
    using FormFactorG = std::map<int, double>;
    using FormFactorJ = std::map<int, FormFactorCoeff>;

    // Calculates the approximate neutron form factor at |q|.
    // q is in reciprocal-lattice Cartesian units without the 2*pi factor. The
    // International Tables coefficients use s = |Q| / (4*pi), with Q in A^-1.
    // Therefore Q = 2*pi*q/a_A and s = |q|/(2*a_A).
    inline double form_factor(
        const jams::Vec<double, 3>& q,
        const double& lattice_parameter_angstrom,
        FormFactorG& g,
        FormFactorJ& j) {
      auto s_sq = pow2(jams::norm(q) / (2.0 * lattice_parameter_angstrom));
      auto ffq = 0.0;
      for (auto l : {0, 2, 4, 6}) {
        double p = (l == 0) ? 1.0 : s_sq;
        ffq += g[l] * p *
               (j[l].A * std::exp(-j[l].a * s_sq) + j[l].B * std::exp(-j[l].b * s_sq) + j[l].C * std::exp(-j[l].c * s_sq) + j[l].D);
      }
      return 0.5 * ffq;
    }

    inline double magnetic_neutron_prefactor_barn() {
      return pow2(0.5 * kNeutronGFactor * kClassicalElectronRadiusMeter) / kBarnSquareMeters;
    }

    inline double neutron_cross_section_barn_mev_scale(
        const double sample_time_ps,
        const int periodogram_length,
        const int periodogram_count,
        const int num_cells) {
      if (periodogram_length <= 0) {
        throw std::runtime_error("neutron cross-section periodogram length must be positive");
      }
      if (periodogram_count <= 0) {
        throw std::runtime_error("neutron cross-section periodogram count must be positive");
      }
      if (num_cells <= 0) {
        throw std::runtime_error("neutron cross-section unit-cell count must be positive");
      }
      return magnetic_neutron_prefactor_barn()
          * (sample_time_ps * static_cast<double>(periodogram_length))
          / (kTwoPi * kHBarIU * static_cast<double>(periodogram_count) * static_cast<double>(num_cells));
    }

    inline double spin_length_from_moment(const double moment_mev_per_tesla) {
      return moment_mev_per_tesla / (kElectronGFactor * kBohrMagnetonIU);
    }
}

#endif  // JAMS_HELPERS_NEUTRON_UNITS_H
