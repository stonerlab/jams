//
// Created by Joe Barker on 2017/09/15.
//

#include <iostream>
#include <cassert>
#include <stdexcept>

#include "jams/core/config.h"

using namespace libconfig;

void config_patch_simple(Setting& orig, const Setting& patch);
void config_patch_element(Setting& orig, const Setting& patch);
void config_patch_aggregate(Setting& orig, const Setting& patch);

void config_patch_simple(Setting& orig, const Setting& patch) {
  if (patch.isAggregate()) {
    config_patch_aggregate(orig, patch);
  } else {
    if (orig.exists(patch.getName())) {
      orig.remove(patch.getName());
    }

    if (patch.getType() == Setting::Type::TypeInt) {
      orig.add(patch.getName(), patch.getType()) = static_cast<int>(patch);
      orig.setFormat(patch.getFormat());
      return;
    }

    if (patch.getType() == Setting::Type::TypeInt64) {
      orig.add(patch.getName(), patch.getType()) = static_cast<int64_t>(patch);
      orig.setFormat(patch.getFormat());
      return;
    }

    if (patch.getType() == Setting::Type::TypeFloat) {
      orig.add(patch.getName(), patch.getType()) = static_cast<double>(patch);
      return;
    }

    if (patch.getType() == Setting::Type::TypeString) {
      orig.add(patch.getName(), patch.getType()) = patch.c_str();
      return;
    }

    if (patch.getType() == Setting::Type::TypeBoolean) {
      orig.add(patch.getName(), patch.getType()) = bool(patch);
      return;
    }

    throw std::runtime_error("Unknown config setting type");
  }
}

Setting& config_patch_add_or_merge_aggregate(Setting &orig, const Setting &patch) {

  // root
  if (patch.getPath().empty()) {
    return orig.lookup(patch.getPath());
  }

  if (orig.exists(patch.getName())) {
    return orig.lookup(patch.getName());
  }

  if (orig.isList() || orig.isArray()) {
    return orig[patch.getIndex()];
  }

  if (patch.getName() == nullptr) {
    return orig.add(patch.getType());
  }

  return orig.add(patch.getName(), patch.getType());
}



void config_patch_aggregate(Setting& orig, const Setting& patch) {

  Setting& aggregate = config_patch_add_or_merge_aggregate(orig, patch);

  const auto length = patch.getLength();
  for (auto i = 0; i < length; ++i) {
    if (patch.isGroup()) {
      config_patch_simple(aggregate, patch[i]);
    } else {
      config_patch_element(aggregate, patch[i]);
    }
  }
}

void config_patch_element(Setting& orig, const Setting& patch) {

  if (patch.isAggregate()) {
    config_patch_aggregate(orig, patch);
    return;
  }

  Setting *setting = nullptr;

  if (orig.getLength() > patch.getIndex()) {
    setting = &orig[patch.getIndex()];
  } else {
    setting = &orig.add(patch.getType());
    setting->setFormat(patch.getFormat());
  }

  if (patch.getType() == Setting::Type::TypeInt) {
    *setting = int(patch);
    return;
  }

  if (patch.getType() == Setting::Type::TypeInt64) {
    *setting = int64_t(patch);
    return;
  }

  if (patch.getType() == Setting::Type::TypeFloat) {
    *setting = double(patch);
    return;
  }

  if (patch.getType() == Setting::Type::TypeString) {
    *setting = patch.c_str();
    return;
  }

  if (patch.getType() == Setting::Type::TypeBoolean) {
    *setting = bool(patch);
    return;
  }

  throw std::runtime_error("unknown setting type");
}

void config_patch(Setting& orig, const Setting& patch) {
  if(!orig.isGroup() && !orig.isList()) {
    return;
  }

  if (patch.isAggregate()) {
    config_patch_aggregate(orig, patch);
  } else {
    config_patch_simple(orig, patch);
  }
}