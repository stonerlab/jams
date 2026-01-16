//
// Created by Joe Barker on 2017/09/15.
//

#ifndef JAMS_CONFIG_H
#define JAMS_CONFIG_H

#include <libconfig.h++>
#include <jams/core/types.h>
#include <jams/helpers/utils.h>
#include <array>
#include <type_traits>


void overwrite_config_settings(libconfig::Setting& orig, const libconfig::Setting& patch);

namespace jams {

namespace detail {
  template<typename>
  struct dependent_false : std::false_type {};
}

template<typename T, typename Enable = void>
struct config_required_impl {
  static T get(const libconfig::Setting &, const std::string &) {
    static_assert(detail::dependent_false<T>::value,
                  "jams::config_required: unsupported type");
  }
};

// Generic implementation for arithmetic types (int/float/double/bool/etc.)
template<typename T>
struct config_required_impl<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  static T get(const libconfig::Setting &setting, const std::string &name) {
    return static_cast<T>(setting[name]);
  }
};

template<typename T>
inline T config_required(const libconfig::Setting &s, const std::string &name) {
  return config_required_impl<T>::get(s, name);
}

// Specialisation for std::array<T, N>
template<typename T, std::size_t N>
struct config_required_impl<std::array<T, N>, void> {
  static std::array<T, N> get(const libconfig::Setting &s, const std::string &name) {
    if (s.getLength() != N)
    {
      throw std::runtime_error("config_required: array size mismatch");
    }
    std::array<T, N> out{};
    const auto &arr = s[name];
    for (std::size_t i = 0; i < N; ++i) {
      out[i] = static_cast<T>(arr[static_cast<int>(i)]);
    }
    return out;
  }
};

// Specialisations for scalar / simple types
template<>
struct config_required_impl<std::string, void> {
  static std::string get(const libconfig::Setting &setting, const std::string &name) {
    return setting[name].c_str();
  }
};

  // Specialisation for nested arrays: std::array<std::array<T, N>, M>
  template<typename T, std::size_t N, std::size_t M>
  struct config_required_impl<std::array<std::array<T, N>, M>, void> {
    static std::array<std::array<T, N>, M> get(const libconfig::Setting &s, const std::string &name)
    {
      std::array<std::array<T, N>, M> out{};
      const auto &mat = s[name];

      for (std::size_t i = 0; i < M; ++i) {
        const auto &row = mat[static_cast<int>(i)];
        for (std::size_t j = 0; j < N; ++j) {
          out[i][j] = static_cast<T>(row[static_cast<int>(j)]);
        }
      }

      return out;
    }
  };

template<>
struct config_required_impl<CoordinateFormat, void> {
  static CoordinateFormat get(const libconfig::Setting &setting, const std::string &name) {
    auto format = jams::config_required<std::string>(setting, name);
    if (lowercase(format) == "fractional") {
      return CoordinateFormat::FRACTIONAL;
    } else if (lowercase(format) == "cartesian") {
      return CoordinateFormat::CARTESIAN;
    } else {
      throw std::runtime_error("Unknown coordinate format");
    }
  }
};

template<>
struct config_required_impl<InteractionFileFormat, void> {
  static InteractionFileFormat get(const libconfig::Setting &setting, const std::string &name) {
    auto format = jams::config_required<std::string>(setting, name);
    if (lowercase(format) == "jams") {
      return InteractionFileFormat::JAMS;
    } else if (lowercase(format) == "kkr") {
      return InteractionFileFormat::KKR;
    } else {
      throw std::runtime_error("Unknown interaction file format");
    }
  }
};

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


}

libconfig::Setting& config_find_setting_by_key_value_pair(const libconfig::Setting& settings, const std::string& key, const std::string& value);

#endif //JAMS_CONFIG_H
