#include <jams/core/base.h>
#include <jams/interface/config.h>

#include <iostream>

Base::Base(const std::string& name, bool verbose, bool debug) :
    name_(name),
    verbose_(verbose),
    debug_(debug)
{
  if (debug_is_enabled()) {
    std::cout << "  DEBUG\n";
  }

  if (verbose_is_enabled()) {
    std::cout << "  VERBOSE\n";
  }
}

Base::Base(const libconfig::Setting &settings)
: Base(jams::config_optional<std::string>(settings, "module", ""),
       jams::config_optional<bool>(settings, "verbose", false),
       jams::config_optional<bool>(settings, "debug", false)) {}