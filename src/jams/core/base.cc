#include <jams/core/base.h>
#include <jams/helpers/utils.h>
#include <jams/interface/config.h>

#include <iostream>
#include <optional>
#include <stdexcept>

namespace {
std::string module_name_from(const libconfig::Setting& settings) {
  return lowercase(jams::config_optional<std::string>(settings, "module", ""));
}

std::optional<std::string> explicit_name_from(const libconfig::Setting& settings) {
  if (settings.exists("name")) {
    auto value = jams::config_optional<std::string>(settings, "name", "");
    if (value.empty()) {
      throw std::runtime_error("configured name must not be empty");
    }
    return value;
  }

  if (settings.exists("id")) {
    auto value = jams::config_optional<std::string>(settings, "id", "");
    if (value.empty()) {
      throw std::runtime_error("configured id must not be empty");
    }
    return value;
  }

  return std::nullopt;
}

int module_occurrence(const libconfig::Setting& settings) {
  auto occurrence = 1;

  try {
    const auto& parent = settings.getParent();
    const auto index = settings.getIndex();
    const auto module_name = module_name_from(settings);

    for (auto i = 0; i < index; ++i) {
      const auto& sibling = parent[i];
      if (module_name_from(sibling) == module_name) {
        ++occurrence;
      }
    }
  } catch (const libconfig::SettingException&) {
  }

  return occurrence;
}

std::string generated_name_from(const libconfig::Setting& settings) {
  const auto module_name = module_name_from(settings);
  const auto occurrence = module_occurrence(settings);
  return occurrence == 1 ? module_name : module_name + "_" + std::to_string(occurrence);
}

std::string instance_name_from(const libconfig::Setting& settings) {
  auto name = explicit_name_from(settings).value_or(generated_name_from(settings));

  try {
    const auto& parent = settings.getParent();
    const auto index = settings.getIndex();

    for (auto i = 0; i < index; ++i) {
      if (instance_name_from(parent[i]) == name) {
        throw std::runtime_error("duplicate configured instance name '" + name + "'");
      }
    }
  } catch (const libconfig::SettingException&) {
  }

  return name;
}
}

Base::Base(const std::string& name, bool verbose, bool debug) :
    module_name_(name),
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
: Base(module_name_from(settings),
       jams::config_optional<bool>(settings, "verbose", false),
       jams::config_optional<bool>(settings, "debug", false)) {
  name_ = instance_name_from(settings);
}
