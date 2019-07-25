//
// Created by Joseph Barker on 2019-07-25.
//

#ifndef JAMS_NEUTRONS_H
#define JAMS_NEUTRONS_H

#include "jams/interface/config.h"

namespace jams {
    struct FormFactorCoeff { double A, a, B, b, C, c, D; };
    using FormFactorG = std::map<int, double>;
    using FormFactorJ = std::map<int, FormFactorCoeff>;

    template<>
    inline FormFactorCoeff config_required(const libconfig::Setting &s, const std::string &name) {
      return FormFactorCoeff{double{s[name][0]}, double{s[name][1]}, double{s[name][2]}, double{s[name][3]},
                             double{s[name][4]}, double{s[name][5]}, double{s[name][6]}};
    }

    template<>
    inline FormFactorG config_required(const libconfig::Setting &s, const std::string &name) {
      FormFactorG g;
      for (auto l : {0,2,4,6}) { g[l] = double{s[name][l/2]}; }
      return g;
    }

    // Calculates the approximate neutron form factor at |q|
    // Approximation and constants from International Tables for Crystallography: Vol. C (pp. 454–461).
    inline double form_factor(const Vec3 &q, const double& lattice_parameter, FormFactorG &g, FormFactorJ &j) {
      auto s_sq = pow2(norm(q) / (4.0 * kPi * lattice_parameter));
      auto ffq = 0.0;
      for (auto l : {0,2,4,6}) {
        double p = (l == 0) ? 1.0 : s_sq;
        ffq += g[l] * p * (j[l].A * exp(-j[l].a * s_sq) + j[l].B * exp(-j[l].b * s_sq) + j[l].C * exp(-j[l].c * s_sq) + j[l].D);
      }
      return 0.5 * ffq;
    }
}

#endif //JAMS_NEUTRONS_H
