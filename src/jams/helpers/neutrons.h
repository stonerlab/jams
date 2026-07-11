//
// Created by Joseph Barker on 2019-07-25.
//

#ifndef JAMS_NEUTRONS_H
#define JAMS_NEUTRONS_H

#include <utility>
#include <vector>

#include "jams/interface/config.h"
#include "jams/helpers/neutron_units.h"

namespace jams {
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

    std::pair<std::vector<FormFactorG>, std::vector<FormFactorJ>> read_form_factor_settings(libconfig::Setting &settings);
}

#endif //JAMS_NEUTRONS_H
