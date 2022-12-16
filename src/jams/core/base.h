//
// Created by Joe Barker on 2017/11/17.
//

#ifndef JAMS_BASE_H
#define JAMS_BASE_H

#include <string>

namespace libconfig { class Setting; }

class Base {
public:
    inline Base() = default;
    explicit Base(const std::string& name, bool verbose = false, bool debug = false);
    explicit Base(const libconfig::Setting& settings);

    inline bool debug_is_enabled() const;
    inline void set_debug(bool value);

    inline bool verbose_is_enabled() const;
    inline void set_verbose(bool value);

    inline const std::string& name() const;
    inline void set_name(const std::string &value);

private:
    std::string name_;
    bool verbose_ = false;
    bool debug_   = false;
};

inline bool Base::debug_is_enabled() const {
  return debug_;
}

inline void Base::set_debug(bool value) {
  debug_ = value;
}

inline bool Base::verbose_is_enabled() const {
  return verbose_ || debug_;
}

inline void Base::set_verbose(bool value) {
  verbose_ = value;
}

inline const std::string &Base::name() const {
  return name_;
}

inline void Base::set_name(const std::string &value) {
  name_ = value;
}


#endif //JAMS_BASE_H
