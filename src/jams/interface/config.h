//
// Created by Joe Barker on 2017/09/15.
//

#ifndef JAMS_CONFIG_H
#define JAMS_CONFIG_H

#include <libconfig.h++>
#include <jams/core/types.h>
#include <jams/helpers/exception.h>
#include <jams/helpers/utils.h>
#include <array>
#include <cstdint>
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
    std::array<T, N> out{};
    const auto &arr = s[name];
    if (arr.getLength() != N) {
      throw std::runtime_error("config_required: array size mismatch");
    }
    for (std::size_t i = 0; i < N; ++i) {
      out[i] = static_cast<T>(arr[static_cast<int>(i)]);
    }
    return out;
  }
};

// Specialisation for jams::Vec<T, N>
template<typename T, std::size_t N>
struct config_required_impl<jams::Vec<T, N>, void> {
  static jams::Vec<T, N> get(const libconfig::Setting &s, const std::string &name) {
    return jams::Vec<T, N>{config_required<std::array<T, N>>(s, name)};
  }
};

// Specialisations for scalar / simple types
template<>
struct config_required_impl<std::string, void> {
  static std::string get(const libconfig::Setting &setting, const std::string &name) {
    return setting[name].c_str();
  }
};

  // Specialisation for nested arrays: std::array<std::array<T, Cols>, Rows>
  template<typename T, std::size_t Cols, std::size_t Rows>
  struct config_required_impl<std::array<std::array<T, Cols>, Rows>, void> {
    static std::array<std::array<T, Cols>, Rows> get(const libconfig::Setting &s, const std::string &name)
    {
      std::array<std::array<T, Cols>, Rows> out{};
      const auto &mat = s[name];

      for (std::size_t i = 0; i < Rows; ++i) {
        const auto &row = mat[static_cast<int>(i)];
        for (std::size_t j = 0; j < Cols; ++j) {
          out[i][j] = static_cast<T>(row[static_cast<int>(j)]);
        }
      }

      return out;
  }
};

// Specialisation for jams::Mat<T, Rows, Cols>
template<typename T, std::size_t Rows, std::size_t Cols>
struct config_required_impl<jams::Mat<T, Rows, Cols>, void> {
  static jams::Mat<T, Rows, Cols> get(const libconfig::Setting &s, const std::string &name) {
    return jams::Mat<T, Rows, Cols>{config_required<std::array<std::array<T, Cols>, Rows>>(s, name)};
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

    // Returns true if the setting is an integer type
    inline bool is_integer_setting(const libconfig::Setting& setting)
    {
      const auto type = setting.getType();
      return type == libconfig::Setting::TypeInt || type == libconfig::Setting::TypeInt64;
    }

    // Returns true if the setting is an array of the requested length, with all entries numeric.
    inline bool is_numeric_array_setting(const libconfig::Setting& setting, const int length)
    {
      if (!setting.isArray() || setting.getLength() != length) {
        return false;
      }

      for (auto i = 0; i < length; ++i) {
        if (!setting[i].isNumber()) {
          return false;
        }
      }

      return true;
    }

    // Returns true if the setting is an array of three numbers
    inline bool is_vec3_setting(const libconfig::Setting& setting)
    {
      return is_numeric_array_setting(setting, 3);
    }

    // Returns an integer setting value. Int64 values are narrowed to int.
    inline int read_integer_setting(const libconfig::Setting& setting, const char* name)
    {
      if (!is_integer_setting(setting)) {
        throw jams::ConfigException(setting, name, " must be an integer");
      }

      if (setting.getType() == libconfig::Setting::TypeInt64) {
        return int(static_cast<int64_t>(setting));
      }

      return int(setting);
    }

    // Returns a numeric setting value converted to T.
    template <typename T>
    inline T read_numeric_setting(const libconfig::Setting& setting, const char* name)
    {
      static_assert(std::is_arithmetic_v<T>, "read_numeric_setting requires an arithmetic type");

      if (!setting.isNumber()) {
        throw jams::ConfigException(setting, name, " must be numeric");
      }

      if (setting.getType() == libconfig::Setting::TypeInt) {
        return T(int(setting));
      }
      if (setting.getType() == libconfig::Setting::TypeInt64) {
        return T(static_cast<int64_t>(setting));
      }

      return T(double(setting));
    }

    // Returns a numeric Vec setting read directly from a positional setting.
    template <typename T, std::size_t N>
    inline jams::Vec<T, N> read_vec_setting(const libconfig::Setting& setting, const char* name)
    {
      if (!is_numeric_array_setting(setting, int(N))) {
        throw jams::ConfigException(setting, name, " must be an array containing exactly ", N, " numeric components");
      }

      jams::Vec<T, N> result;
      for (auto i = 0; i < int(N); ++i) {
        result[i] = read_numeric_setting<T>(setting[i], "component");
      }
      return result;
    }

    // Returns a row-major numeric Mat setting read directly from a flat positional setting.
    template <typename T, std::size_t Rows, std::size_t Cols>
    inline jams::Mat<T, Rows, Cols> read_mat_setting(const libconfig::Setting& setting, const char* name)
    {
      constexpr auto length = int(Rows * Cols);
      if (!is_numeric_array_setting(setting, length)) {
        throw jams::ConfigException(setting, name, " must be an array containing exactly ", length, " numeric components");
      }

      jams::Mat<T, Rows, Cols> result;
      for (auto row = 0; row < int(Rows); ++row) {
        for (auto col = 0; col < int(Cols); ++col) {
          result[row][col] = read_numeric_setting<T>(setting[row * int(Cols) + col], "component");
        }
      }
      return result;
    }





}

libconfig::Setting& config_find_setting_by_key_value_pair(const libconfig::Setting& settings, const std::string& key, const std::string& value);

#endif //JAMS_CONFIG_H
