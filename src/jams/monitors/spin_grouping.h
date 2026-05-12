// spin_grouping.h                                                    -*-C++-*-

#ifndef JAMS_MONITORS_SPIN_GROUPING_H
#define JAMS_MONITORS_SPIN_GROUPING_H

#include <jams/containers/multiarray.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/interface/config.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace jams::monitors {

enum class SpinGrouping {
  NONE,
  MATERIALS,
  POSITIONS
};

struct SpinGroup {
  std::string name;
  jams::MultiArray<int, 1> indices;
};

inline SpinGrouping parse_spin_grouping(
    const libconfig::Setting& settings,
    const std::string& default_grouping,
    const std::string& quantity_name) {
  const auto grouping_str = lowercase(
      jams::config_optional<std::string>(settings, "grouping", default_grouping));

  if (grouping_str == "none" || grouping_str == "total") {
    return SpinGrouping::NONE;
  }

  if (grouping_str == "material" || grouping_str == "materials") {
    return SpinGrouping::MATERIALS;
  }

  if (grouping_str == "position" || grouping_str == "positions"
      || grouping_str == "unit-cell-positions"
      || grouping_str == "unit_cell_positions") {
    return SpinGrouping::POSITIONS;
  }

  throw std::runtime_error("unknown " + quantity_name + " grouping: " + grouping_str);
}

inline std::vector<SpinGroup> make_spin_groups(const SpinGrouping grouping) {
  std::vector<SpinGroup> groups;

  switch (grouping) {
    case SpinGrouping::NONE: {
      jams::MultiArray<int, 1> indices(globals::num_spins);
      for (auto i = 0; i < globals::num_spins; ++i) {
        indices(i) = i;
      }
      groups.push_back({"total", indices});
      break;
    }

    case SpinGrouping::MATERIALS: {
      std::vector<std::vector<int>> material_index_groups(globals::lattice->num_materials());
      for (auto i = 0; i < globals::num_spins; ++i) {
        const auto type = globals::lattice->lattice_site_material_id(i);
        material_index_groups[type].push_back(i);
      }

      groups.reserve(material_index_groups.size());
      for (std::size_t n = 0; n < material_index_groups.size(); ++n) {
        groups.push_back({
            globals::lattice->material_name(static_cast<int>(n)),
            jams::MultiArray<int, 1>(
                material_index_groups[n].begin(),
                material_index_groups[n].end())});
      }
      break;
    }

    case SpinGrouping::POSITIONS: {
      std::vector<std::vector<int>> position_index_groups(globals::lattice->num_basis_sites());
      for (auto i = 0; i < globals::num_spins; ++i) {
        const auto pos = globals::lattice->lattice_site_basis_index(i);
        position_index_groups[pos].push_back(i);
      }

      groups.reserve(position_index_groups.size());
      for (std::size_t n = 0; n < position_index_groups.size(); ++n) {
        const auto mat = globals::lattice->basis_site_atom(static_cast<int>(n)).material_index;
        const auto material_name = globals::lattice->material_name(mat);
        groups.push_back({
            material_name + "_" + std::to_string(n + 1),
            jams::MultiArray<int, 1>(
                position_index_groups[n].begin(),
                position_index_groups[n].end())});
      }
      break;
    }
  }

  return groups;
}

inline std::string grouped_column_name(
    const SpinGrouping grouping,
    const std::string& group_name,
    const std::string& name) {
  if (grouping == SpinGrouping::NONE) {
    return name;
  }

  return group_name + "_" + name;
}

}  // namespace jams::monitors

#endif  // JAMS_MONITORS_SPIN_GROUPING_H
