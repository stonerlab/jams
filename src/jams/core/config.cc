//
// Created by Joe Barker on 2017/09/15.
//

#include "jams/core/config.h"
#include <iostream>
#include <cassert>

using namespace libconfig;

Setting&  add_settings(Setting& original, const Setting& replace, const bool& allow_type_change);
void replace_settings(Setting& original, const Setting& replace, const bool& allow_type_change);

Setting& add_float(Setting& original, const Setting& addition) {
  if (addition.getName() == NULL) {
    return original.add(addition.getType()) = double(addition);
  } else {
    return original.add(addition.getName(), addition.getType()) = double(addition);
  }
}

Setting& add_int(Setting& original, const Setting& addition) {
  if (addition.getName() == NULL) {
    return original.add(addition.getType()) = int(addition);
  } else {
    return original.add(addition.getName(), addition.getType()) = int(addition);
  }
}

Setting& add_int64(Setting& original, const Setting& addition) {
  if (addition.getName() == NULL) {
    return original.add(addition.getType()) = int64_t(addition);
  } else {
    return original.add(addition.getName(), addition.getType()) = int64_t(addition);
  }
}

Setting& add_string(Setting& original, const Setting& addition) {
  if (addition.getName() == NULL) {
    return original.add(addition.getType()) = addition.c_str();
  } else {
    return original.add(addition.getName(), addition.getType()) = addition.c_str();
  }
}

Setting& add_boolean(Setting& original, const Setting& addition) {
  if (addition.getName() == NULL) {
    return original.add(addition.getType()) = bool(addition);
  } else {
    return original.add(addition.getName(), addition.getType()) = bool(addition);
  }
}

Setting& add_array(Setting& original, const Setting& addition, const bool& allow_type_change) {
  std::cerr << "-array-" << std::endl;
  if (addition.getName() == NULL) {
    Setting &new_setting = original.add(Setting::Type::TypeArray);
    for (auto i = 0; i < addition.getLength(); ++i) {
      add_settings(original, addition[i], allow_type_change);
    }
    return new_setting;
  } else {
    Setting &new_setting = original.add(addition.getName(), Setting::Type::TypeArray);
    for (auto i = 0; i < addition.getLength(); ++i) {
      add_settings(original, addition[i], allow_type_change);
    }
  }
}

Setting& add_list(Setting& original, const Setting& addition, const bool& allow_type_change) {
  std::cerr << "-list-" << std::endl;
  auto name = addition.getName();
  if (name == nullptr) {
    Setting& new_setting = original.add(Setting::Type::TypeList);
    for (auto i = 0; i < addition.getLength(); ++i) {
      replace_settings(original, addition[i], allow_type_change);
    }
    return new_setting;
  } else {
    Setting& new_setting = original.add(addition.getName(), Setting::Type::TypeList);
    for (auto i = 0; i < addition.getLength(); ++i) {
      replace_settings(original, addition[i], allow_type_change);
    }
    return new_setting;
  }
}

Setting& add_group(Setting& original, const Setting& addition, const bool& allow_type_change) {
  std::cerr << "-group-" << std::endl;
  auto name = addition.getName();
  if (name == nullptr) {
    Setting& new_setting = original.add(Setting::Type::TypeGroup);
    for (auto i = 0; i < addition.getLength(); ++i) {
      replace_settings(original, addition[i], allow_type_change);
    }
    return new_setting;
  } else {
    Setting& new_setting = original.add(addition.getName(), Setting::Type::TypeGroup);
    for (auto i = 0; i < addition.getLength(); ++i) {
      replace_settings(original, addition[i], allow_type_change);
    }
    return new_setting;
  }
}

Setting& add_settings(Setting& original, const Setting& replace, const bool& allow_type_change) {
  auto type = replace.getType();

  if (type == Setting::Type::TypeFloat) {
    return add_float(original, replace);
  }

  if (type == Setting::Type::TypeInt) {
    return add_int(original, replace);
  }

  if (type == Setting::Type::TypeInt64) {
    return add_int64(original, replace);
  }

  if (type == Setting::Type::TypeString) {
    return add_string(original, replace);
  }

  if (type == Setting::Type::TypeBoolean) {
    return add_boolean(original, replace);
  }

  if (type == Setting::Type::TypeGroup) {
    return add_group(original, replace, allow_type_change);
  }

  if (type == Setting::Type::TypeList) {
    return add_list(original, replace, allow_type_change);
  }

  if (type == Setting::Type::TypeArray) {
    return add_array(original, replace, allow_type_change);
  }

  throw std::runtime_error("unknown setting type");
}

void replace_settings(Setting& original, const Setting& replace, const bool& allow_type_change) {
  auto num_settings = replace.getLength();

  for (auto i = 0; i < num_settings; ++i) {
    Setting& sub_setting = replace[i];
    auto name = sub_setting.getName();
    auto path = sub_setting.getPath();
    auto type = sub_setting.getType();

    if (sub_setting.isGroup() || sub_setting.isList()) {
      if (original.exists(path)) {
        replace_settings(original, sub_setting, allow_type_change);
      } else {
        add_settings(original, sub_setting, allow_type_change);
      }
    } else {
      std::cout << path << "\t" << original.exists(path) << std::endl;
      if (original.exists(path)) {
        if (!allow_type_change && type != original.lookup(path).getType()) {
          throw SettingTypeException(replace[i]);
        }
        Setting& parent = original.lookup(path).getParent();
        parent.remove(name);
        add_settings(parent, sub_setting, allow_type_change);
      }
    }
  }
}