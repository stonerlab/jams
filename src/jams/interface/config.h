//
// Created by Joe Barker on 2017/09/15.
//

#ifndef JAMS_CONFIG_H
#define JAMS_CONFIG_H

#include <libconfig.h++>
#include <jams/helpers/utils.h>
#include "jams/core/types.h"

void config_patch(libconfig::Setting& orig, const libconfig::Setting& patch);

namespace jams {

    template<typename T>
    inline T config_required(const libconfig::Setting &setting, const std::string &name);

    template<typename T>
    inline T config_optional(const libconfig::Setting &setting, const std::string &name, const T& def) {
        if (setting.exists(name)) {
          return config_required<T>(setting, name);
        } else {
          return def;
        }
    }

    template<>
    inline std::string config_required(const libconfig::Setting &setting, const std::string &name) {
      return setting[name].c_str();
    }

    template<>
    inline bool config_required(const libconfig::Setting &setting, const std::string &name) {
      return bool(setting[name]);
    }

    template<>
    inline int config_required(const libconfig::Setting &setting, const std::string &name) {
      return int(setting[name]);
    }

    template<>
    inline long config_required(const libconfig::Setting &setting, const std::string &name) {
      return long(setting[name]);
    }

    template<>
    inline double config_required(const libconfig::Setting &setting, const std::string &name) {
      return double(setting[name]);
    }

    template<>
    inline Vec3 config_required(const libconfig::Setting &setting, const std::string &name) {
      return {double(setting[name][0]), double(setting[name][1]), double(setting[name][2])};
    }

    template<>
    inline Vec3b config_required(const libconfig::Setting &setting, const std::string &name) {
      return {bool(setting[name][0]), bool(setting[name][1]), bool(setting[name][2])};
    }

    template<>
    inline Vec3i config_required(const libconfig::Setting &setting, const std::string &name) {
      return {int(setting[name][0]), int(setting[name][1]), int(setting[name][2])};
    }

    template<>
    inline Mat3 config_required(const libconfig::Setting &setting, const std::string &name) {
      return {setting[name][0][0], setting[name][0][1], setting[name][0][2],
              setting[name][1][0], setting[name][1][1], setting[name][1][2],
              setting[name][2][0], setting[name][2][1], setting[name][2][2]};
      }

    inline CoordinateFormat config_required(const libconfig::Setting &setting, const std::string &name) {
      auto format = jams::config_required<string>(setting, "coordinate_format");
      if (lowercase(format) == "fractional") {
        return CoordinateFormat::Fractional;
      } else if (lowercase(format) == "fractional") {
        return CoordinateFormat::Cartesian;
      } else {
        throw std::runtime_error("Unknown coordinate format");
      }
    }

}


#endif //JAMS_CONFIG_H
