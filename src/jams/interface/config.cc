//
// Created by Joe Barker on 2017/09/15.
//

#include <iostream>
#include <cassert>
#include <stdexcept>

#include "config.h"

void config_patch_simple(libconfig::Setting& orig, const libconfig::Setting& patch);
void config_patch_element(libconfig::Setting& orig, const libconfig::Setting& patch);
void config_patch_aggregate(libconfig::Setting& orig, const libconfig::Setting& patch);

void config_patch_simple(libconfig::Setting& orig, const libconfig::Setting& patch) {
  if (patch.isAggregate()) {
    config_patch_aggregate(orig, patch);
  } else {
    if (orig.exists(patch.getName())) {
      orig.remove(patch.getName());
    }

    if (patch.getType() == libconfig::Setting::Type::TypeInt) {
      orig.add(patch.getName(), patch.getType()) = static_cast<int>(patch);
      orig.setFormat(patch.getFormat());
      return;
    }

    if (patch.getType() == libconfig::Setting::Type::TypeInt64) {
      orig.add(patch.getName(), patch.getType()) = static_cast<int64_t>(patch);
      orig.setFormat(patch.getFormat());
      return;
    }

    if (patch.getType() == libconfig::Setting::Type::TypeFloat) {
      orig.add(patch.getName(), patch.getType()) = static_cast<double>(patch);
      return;
    }

    if (patch.getType() == libconfig::Setting::Type::TypeString) {
      orig.add(patch.getName(), patch.getType()) = patch.c_str();
      return;
    }

    if (patch.getType() == libconfig::Setting::Type::TypeBoolean) {
      orig.add(patch.getName(), patch.getType()) = bool(patch);
      return;
    }

    throw std::runtime_error("Unknown config setting type");
  }
}

libconfig::Setting& config_patch_add_or_merge_aggregate(libconfig::Setting &orig, const libconfig::Setting &patch) {

  // root
  if (patch.getPath().empty()) {
    return orig;
  }

  if (orig.exists(patch.getName())) {
    return orig.lookup(patch.getName());
  }

  if (orig.isList() || orig.isArray()) {
    // the origin's length is shorter then we need to make them the same
    // so we can access all of the indicies. We have to be a little
    // careful to get the types correct for each element because lists
    // contain elements of different types.
    if (orig.getLength() < patch.getParent().getLength()) {
      for (auto i = orig.getLength(); i < patch.getParent().getLength(); ++i) {
        orig.add(patch.getParent()[i].getType());
      }
    }
    return orig[patch.getIndex()];
  }

  if (patch.getName() == nullptr) {
    return orig.add(patch.getType());
  }

  return orig.add(patch.getName(), patch.getType());
}



void config_patch_aggregate(libconfig::Setting& orig, const libconfig::Setting& patch) {

  libconfig::Setting& aggregate = config_patch_add_or_merge_aggregate(orig, patch);

  const auto length = patch.getLength();
  for (auto i = 0; i < length; ++i) {
    if (patch.isGroup()) {
      config_patch_simple(aggregate, patch[i]);
    } else {
      config_patch_element(aggregate, patch[i]);
    }
  }
}

void config_patch_element(libconfig::Setting& orig, const libconfig::Setting& patch) {

  if (patch.isAggregate()) {
    config_patch_aggregate(orig, patch);
    return;
  }

  libconfig::Setting *setting = nullptr;

  if (orig.getLength() > patch.getIndex()) {
    setting = &orig[patch.getIndex()];
  } else {
    setting = &orig.add(patch.getType());
    setting->setFormat(patch.getFormat());
  }

  if (patch.getType() == libconfig::Setting::Type::TypeInt) {
    *setting = int(patch);
    return;
  }

  if (patch.getType() == libconfig::Setting::Type::TypeInt64) {
    *setting = int64_t(patch);
    return;
  }

  if (patch.getType() == libconfig::Setting::Type::TypeFloat) {
    *setting = double(patch);
    return;
  }

  if (patch.getType() == libconfig::Setting::Type::TypeString) {
    *setting = patch.c_str();
    return;
  }

  if (patch.getType() == libconfig::Setting::Type::TypeBoolean) {
    *setting = bool(patch);
    return;
  }

  throw std::runtime_error("unknown setting type");
}

void overwrite_config_settings(libconfig::Setting& orig, const libconfig::Setting& patch) {
  if(!orig.isGroup() && !orig.isList()) {
    return;
  }

  if (patch.isAggregate()) {
    config_patch_aggregate(orig, patch);
  } else {
    config_patch_simple(orig, patch);
  }
}


libconfig::Setting& config_find_setting_by_key_value_pair(const libconfig::Setting& settings, const std::string& key, const std::string& value) {
  for (auto i = 0; i < settings.getLength(); ++i) {
    std::string module_name = settings[i][key].c_str();
    if (module_name == value) {
      return settings[i];
    }
  }

  const std::string error_string = key + "=" + value;
  throw libconfig::SettingNotFoundException(settings, error_string.c_str());
}