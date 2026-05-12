// spin_grouping.h                                                    -*-C++-*-

#ifndef JAMS_MONITORS_SPIN_GROUPING_H
#define JAMS_MONITORS_SPIN_GROUPING_H

#include <jams/containers/multiarray.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/interface/config.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jams::monitors {

enum class SpinGrouping {
  NONE,
  MATERIALS,
  POSITIONS
};

struct SpinGroup {
  using index_array_type = jams::MultiArray<int, 1>;
  using index_span_type = typename index_array_type::const_span_type;
  using size_type = typename index_array_type::size_type;

  std::string name;
  index_array_type indices;

  [[nodiscard]] index_span_type indices_span() const {
    return indices.host_span();
  }

  [[nodiscard]] const index_array_type& indices_array() const noexcept {
    return indices;
  }

  [[nodiscard]] bool empty() const noexcept {
    return indices.empty();
  }

  [[nodiscard]] size_type size() const noexcept {
    return indices.size();
  }
};

namespace detail {

inline constexpr std::array<std::string_view, 2> none_grouping_names = {
    "none",
    "total"};

inline constexpr std::array<std::string_view, 2> material_grouping_names = {
    "material",
    "materials"};

inline constexpr std::array<std::string_view, 4> position_grouping_names = {
    "position",
    "positions",
    "unit-cell-positions",
    "unit_cell_positions"};

template<std::size_t N>
[[nodiscard]] inline bool matches_grouping_name(
    const std::string_view name,
    const std::array<std::string_view, N>& aliases) {
  return std::ranges::any_of(aliases, [name](const std::string_view alias) {
    return name == alias;
  });
}

template<std::integral Index>
[[nodiscard]] inline std::size_t checked_group_index(
    const Index index,
    const std::size_t num_groups,
    const std::string_view grouping_name) {
  if (std::cmp_less(index, 0) || std::cmp_greater_equal(index, num_groups)) {
    throw std::runtime_error("invalid " + std::string(grouping_name) + " spin group index");
  }

  return static_cast<std::size_t>(index);
}

template<class GroupIndexFn, class GroupNameFn>
[[nodiscard]] inline std::vector<SpinGroup> make_indexed_spin_groups(
    const std::size_t num_groups,
    GroupIndexFn group_index_for_spin,
    GroupNameFn group_name,
    const std::string_view grouping_name) {
  std::vector<std::size_t> group_sizes(num_groups, 0);
  for (const int spin_index : std::views::iota(0, globals::num_spins)) {
    const auto group_index = checked_group_index(
        group_index_for_spin(spin_index),
        num_groups,
        grouping_name);
    ++group_sizes[group_index];
  }

  std::vector<SpinGroup> groups;
  groups.reserve(num_groups);
  for (std::size_t group_index = 0; group_index < num_groups; ++group_index) {
    groups.push_back({
        std::string(group_name(group_index)),
        SpinGroup::index_array_type(group_sizes[group_index])});
  }

  std::vector<std::size_t> group_offsets(num_groups, 0);
  for (const int spin_index : std::views::iota(0, globals::num_spins)) {
    const auto group_index = checked_group_index(
        group_index_for_spin(spin_index),
        num_groups,
        grouping_name);
    auto indices = groups[group_index].indices.mutable_host_span();
    indices[group_offsets[group_index]++] = spin_index;
  }

  return groups;
}

}  // namespace detail

[[nodiscard]]
inline SpinGrouping parse_spin_grouping(
    const libconfig::Setting& settings,
    const std::string_view default_grouping,
    const std::string_view quantity_name) {
  const auto grouping_str = lowercase(
      jams::config_optional<std::string>(settings, "grouping", std::string(default_grouping)));

  if (detail::matches_grouping_name(grouping_str, detail::none_grouping_names)) {
    return SpinGrouping::NONE;
  }

  if (detail::matches_grouping_name(grouping_str, detail::material_grouping_names)) {
    return SpinGrouping::MATERIALS;
  }

  if (detail::matches_grouping_name(grouping_str, detail::position_grouping_names)) {
    return SpinGrouping::POSITIONS;
  }

  throw std::runtime_error("unknown " + std::string(quantity_name) + " grouping: " + grouping_str);
}

[[nodiscard]]
inline std::vector<SpinGroup> make_spin_groups(const SpinGrouping grouping) {
  std::vector<SpinGroup> groups;

  switch (grouping) {
    case SpinGrouping::NONE: {
      groups.push_back({
          "total",
          SpinGroup::index_array_type(std::views::iota(0, globals::num_spins))});
      break;
    }

    case SpinGrouping::MATERIALS: {
      groups = detail::make_indexed_spin_groups(
          static_cast<std::size_t>(globals::lattice->num_materials()),
          [](const int spin_index) {
            return globals::lattice->lattice_site_material_id(spin_index);
          },
          [](const std::size_t group_index) {
            return globals::lattice->material_name(static_cast<int>(group_index));
          },
          "material");
      break;
    }

    case SpinGrouping::POSITIONS: {
      groups = detail::make_indexed_spin_groups(
          static_cast<std::size_t>(globals::lattice->num_basis_sites()),
          [](const int spin_index) {
            return globals::lattice->lattice_site_basis_index(spin_index);
          },
          [](const std::size_t group_index) {
            const auto material = globals::lattice->basis_site_atom(static_cast<int>(group_index)).material_index;
            return globals::lattice->material_name(material) + "_" + std::to_string(group_index + 1);
          },
          "position");
      break;
    }
  }

  return groups;
}

[[nodiscard]]
inline std::string grouped_column_name(
    const SpinGrouping grouping,
    const std::string_view group_name,
    const std::string_view name) {
  if (grouping == SpinGrouping::NONE) {
    return std::string(name);
  }

  return std::string(group_name) + "_" + std::string(name);
}

}  // namespace jams::monitors

#endif  // JAMS_MONITORS_SPIN_GROUPING_H
