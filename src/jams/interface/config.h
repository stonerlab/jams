//
// Created by Joe Barker on 2017/09/15.
//

#ifndef JAMS_CONFIG_H
#define JAMS_CONFIG_H

#include <libconfig.h++>
#include <jams/helpers/utils.h>
#include <jams/core/interactions.h>
#include "jams/core/types.h"

void overwrite_config_settings(libconfig::Setting& orig, const libconfig::Setting& patch);

namespace jams {

    /// Returns the value of the setting `name` within the group of settings
    /// `s`. If the setting is not found a runtime_error exception is thrown.
    ///
    /// Usually the template parameter type `T` should be specified to ensure
    /// the setting is converted to the correct type in ambigious cases (e.g.
    /// int vs float).
    ///
    /// @example
    /// auto my_param = config_required<double>(some_settings, "value");
    ///
    template<typename T>
    inline T config_required(const libconfig::Setting &s, const std::string &name);

    /// Returns the value of the setting `name` within the group of settings
    /// `s`. If the setting is not found the default value `def` is returned.
    ///
    /// Usually the template parameter type `T` should be specified to ensure
    /// the setting is converted to the correct type in ambigious cases (e.g.
    /// int vs float).
    ///
    /// @example
    /// auto my_param = config_optional<double>(some_settings, "value", 1.0);
    ///
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
    inline unsigned config_required(const libconfig::Setting &setting, const std::string &name) {
      return unsigned(setting[name]);
    }

    template<>
    inline unsigned long config_required(const libconfig::Setting &setting, const std::string &name) {
      return (unsigned long)(setting[name]);
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

    template<>
    inline CoordinateFormat config_required(const libconfig::Setting &setting, const std::string &name) {
      auto format = jams::config_required<std::string>(setting, "coordinate_format");
      if (lowercase(format) == "fractional") {
        return CoordinateFormat::FRACTIONAL;
      } else if (lowercase(format) == "cartesian") {
        return CoordinateFormat::CARTESIAN;
      } else {
        throw std::runtime_error("Unknown coordinate format");
      }
    }

    template<>
    inline InteractionFileFormat config_required(const libconfig::Setting &setting, const std::string &name) {
      auto format = jams::config_required<std::string>(setting, name);
      if (lowercase(format) == "jams") {
        return InteractionFileFormat::JAMS;
      } else if (lowercase(format) == "kkr") {
        return InteractionFileFormat::KKR;
      } else {
        throw std::runtime_error("Unknown interaction file format");
      }
    }

}

libconfig::Setting& config_find_setting_by_key_value_pair(const libconfig::Setting& settings, const std::string& key, const std::string& value);

#endif //JAMS_CONFIG_H
