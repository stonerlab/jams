// Copyright 2014 Joseph Barker. All rights reserved.

#include <jams/initializer/init_damping_regions.h>

#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <libconfig.h++>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/spatial_region.h>
#include <jams/interface/config.h>

namespace {

struct AlphaProfile {
  double low = 0.0;
  double high = 0.0;

  [[nodiscard]] double value(const double u) const {
    return low + u * (high - low);
  }
};

enum class AlphaMode {
  Scalar,
  Materials,
  UnitCellPositions,
};

struct RegionAlpha {
  AlphaMode mode = AlphaMode::Scalar;
  AlphaProfile scalar;
  std::map<int, AlphaProfile> materials;
  std::map<int, AlphaProfile> unit_cell_positions;
};

bool has_any_alpha_value_field(const libconfig::Setting& setting) {
  return setting.exists("alpha") || setting.exists("low") || setting.exists("high");
}

void require_finite_nonnegative_alpha(const libconfig::Setting& setting,
                                      const char* name,
                                      const double alpha) {
  if (!std::isfinite(alpha) || alpha < 0.0) {
    throw jams::ConfigException(
        setting, name, " must be a finite non-negative damping value");
  }
}

AlphaProfile parse_alpha_profile(const libconfig::Setting& setting,
                                 const jams::SpatialRegion& region) {
  if (region.is_constant()) {
    if (setting.exists("low") || setting.exists("high")) {
      throw jams::ConfigException(setting, "constant damping regions use alpha, not low/high");
    }

    if (!setting.exists("alpha")) {
      throw jams::ConfigException(setting, "constant damping regions require alpha");
    }

    const auto alpha = jams::config_required<double>(setting, "alpha");
    require_finite_nonnegative_alpha(setting["alpha"], "alpha", alpha);
    return {alpha, alpha};
  }

  if (setting.exists("alpha")) {
    throw jams::ConfigException(setting, "linear damping regions use low/high, not alpha");
  }

  if (!setting.exists("low") || !setting.exists("high")) {
    throw jams::ConfigException(setting, "linear damping regions require low and high");
  }

  const auto low = jams::config_required<double>(setting, "low");
  const auto high = jams::config_required<double>(setting, "high");
  require_finite_nonnegative_alpha(setting["low"], "low", low);
  require_finite_nonnegative_alpha(setting["high"], "high", high);
  return {low, high};
}

void require_region_list(const libconfig::Setting& setting, const char* name) {
  if (!setting.isList() || setting.getLength() == 0) {
    throw jams::ConfigException(setting, name, " must be a non-empty list");
  }
}

std::map<int, AlphaProfile> parse_material_profiles(
    const libconfig::Setting& settings,
    const jams::SpatialRegion& region) {
  require_region_list(settings, "materials");

  std::map<int, AlphaProfile> profiles;
  std::set<std::string> seen_names;
  for (auto n = 0; n < settings.getLength(); ++n) {
    const auto& entry = settings[n];
    if (!entry.isGroup()) {
      throw jams::ConfigException(entry, "material damping entry", " must be a group");
    }

    const auto material_name = jams::config_required<std::string>(entry, "name");
    if (!globals::lattice->material_exists(material_name)) {
      throw jams::ConfigException(entry["name"], "unknown material '", material_name, "'");
    }

    if (!seen_names.insert(material_name).second) {
      throw jams::ConfigException(
          entry["name"], "duplicate material damping entry for '", material_name, "'");
    }

    profiles.emplace(
        globals::lattice->material_index(material_name),
        parse_alpha_profile(entry, region));
  }

  return profiles;
}

std::map<int, AlphaProfile> parse_unit_cell_position_profiles(
    const libconfig::Setting& settings,
    const jams::SpatialRegion& region) {
  require_region_list(settings, "unit_cell_positions");

  std::map<int, AlphaProfile> profiles;
  std::set<int> seen_indices;
  const auto num_basis_sites = globals::lattice->num_basis_sites();
  for (auto n = 0; n < settings.getLength(); ++n) {
    const auto& entry = settings[n];
    if (!entry.isGroup()) {
      throw jams::ConfigException(entry, "unit cell position damping entry", " must be a group");
    }

    const auto one_based_index = jams::read_integer_setting(entry["index"], "index");
    if (one_based_index < 1 || one_based_index > num_basis_sites) {
      throw jams::ConfigException(
          entry["index"],
          "unit cell position index ",
          one_based_index,
          " is outside [1, ",
          num_basis_sites,
          "]");
    }

    if (!seen_indices.insert(one_based_index).second) {
      throw jams::ConfigException(
          entry["index"],
          "duplicate unit cell position damping entry for index ",
          one_based_index);
    }

    profiles.emplace(one_based_index - 1, parse_alpha_profile(entry, region));
  }

  return profiles;
}

RegionAlpha parse_region_alpha(const libconfig::Setting& setting,
                               const jams::SpatialRegion& region) {
  const bool has_scalar = has_any_alpha_value_field(setting);
  const bool has_materials = setting.exists("materials");
  const bool has_unit_cell_positions = setting.exists("unit_cell_positions");
  const int mode_count =
      int(has_scalar) + int(has_materials) + int(has_unit_cell_positions);

  if (mode_count != 1) {
    throw jams::ConfigException(
        setting,
        "damping region must define exactly one of scalar alpha values, materials, "
        "or unit_cell_positions");
  }

  RegionAlpha alpha;
  if (has_scalar) {
    alpha.mode = AlphaMode::Scalar;
    alpha.scalar = parse_alpha_profile(setting, region);
  } else if (has_materials) {
    alpha.mode = AlphaMode::Materials;
    alpha.materials = parse_material_profiles(setting["materials"], region);
  } else {
    alpha.mode = AlphaMode::UnitCellPositions;
    alpha.unit_cell_positions =
        parse_unit_cell_position_profiles(setting["unit_cell_positions"], region);
  }

  return alpha;
}

std::optional<AlphaProfile> alpha_profile_for_spin(const RegionAlpha& alpha,
                                                   const int spin) {
  switch (alpha.mode) {
    case AlphaMode::Scalar:
      return alpha.scalar;
    case AlphaMode::Materials: {
      const auto material = globals::lattice->lattice_site_material_id(spin);
      const auto it = alpha.materials.find(material);
      if (it == alpha.materials.end()) {
        return std::nullopt;
      }
      return it->second;
    }
    case AlphaMode::UnitCellPositions: {
      const auto unit_cell_position =
          static_cast<int>(globals::lattice->lattice_site_basis_index(spin));
      const auto it = alpha.unit_cell_positions.find(unit_cell_position);
      if (it == alpha.unit_cell_positions.end()) {
        return std::nullopt;
      }
      return it->second;
    }
  }

  return std::nullopt;
}

jams::Vec<double, 3> spin_position(const int spin) {
  return {
      static_cast<double>(globals::positions(spin, 0)),
      static_cast<double>(globals::positions(spin, 1)),
      static_cast<double>(globals::positions(spin, 2)),
  };
}

}  // namespace

void jams::InitDampingRegions::execute(const libconfig::Setting& settings) {
  if (globals::lattice == nullptr) {
    throw jams::ConfigException(
        settings, "damping-regions initializer requires an initialized lattice");
  }

  if (!settings.exists("regions")) {
    throw jams::ConfigException(settings, "damping-regions initializer requires regions");
  }

  const auto& regions = settings["regions"];
  require_region_list(regions, "regions");

  std::vector<int> alpha_assignment(globals::num_spins, -1);

  for (auto region_index = 0; region_index < regions.getLength(); ++region_index) {
    const auto& region_setting = regions[region_index];
    if (!region_setting.isGroup()) {
      throw jams::ConfigException(region_setting, "damping region", " must be a group");
    }

    const auto spatial_region = jams::SpatialRegion::from_config(region_setting);
    const auto region_alpha = parse_region_alpha(region_setting, spatial_region);

    int assigned_spin_count = 0;
    for (auto spin = 0; spin < globals::num_spins; ++spin) {
      const auto position = spin_position(spin);
      if (!spatial_region.contains(position)) {
        continue;
      }

      const auto alpha_profile = alpha_profile_for_spin(region_alpha, spin);
      if (!alpha_profile.has_value()) {
        continue;
      }

      if (alpha_assignment[spin] >= 0) {
        std::ostringstream message;
        message << "damping region " << region_index
                << " assigns spin " << spin
                << " already assigned by region " << alpha_assignment[spin];
        throw jams::ConfigException(region_setting, message.str());
      }

      alpha_assignment[spin] = region_index;
      ++assigned_spin_count;

      const auto u = spatial_region.interpolation_fraction(position);
      globals::alpha(spin) = static_cast<jams::Real>(alpha_profile->value(u));
    }

    if (assigned_spin_count == 0) {
      std::ostringstream message;
      message << "damping region " << region_index << " does not assign any spins";
      throw jams::ConfigException(region_setting, message.str());
    }
  }
}
