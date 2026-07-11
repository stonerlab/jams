#include "jams/helpers/neutrons.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"

#include <stdexcept>

namespace jams {
    std::pair<std::vector<FormFactorG>, std::vector<FormFactorJ>> read_form_factor_settings(libconfig::Setting &settings) {
      auto num_materials = globals::lattice->num_materials();

      if (settings.getLength() != num_materials) {
        throw std::runtime_error("there must be one form factor per material\"");
      }

      std::vector<FormFactorG> g_params(num_materials);
      std::vector<FormFactorJ> j_params(num_materials);

      for (auto i = 0; i < settings.getLength(); ++i) {
        for (auto l : {0,2,4,6}) {
          j_params[i][l] = config_optional<FormFactorCoeff>(settings[i], "j" + std::to_string(l), j_params[i][l]);
        }
        g_params[i] = config_required<FormFactorG>(settings[i], "g");
      }

      return {g_params, j_params};
    }

}
