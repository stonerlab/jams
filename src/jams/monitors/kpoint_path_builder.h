//
// Created by Codex on 2026-02-16.
//

#ifndef JAMS_KPOINT_PATH_BUILDER_H
#define JAMS_KPOINT_PATH_BUILDER_H

#include <jams/core/types.h>
#include <jams/monitors/hkl_index.h>

#include <libconfig.h++>

#include <vector>

class Lattice;

/// @brief Helper to construct k-point paths and segment offsets.
class KPointPathBuilder {
public:
  explicit KPointPathBuilder(Lattice& lattice);

  void append_full_k_grid(
      std::vector<jams::HKLIndex>& k_points,
      std::vector<int>& k_segment_offsets,
      const jams::Vec<int, 3>& kspace_size) const;

  void append_k_path_segment(
      std::vector<jams::HKLIndex>& k_points,
      std::vector<int>& k_segment_offsets,
      libconfig::Setting& settings,
      const jams::Vec<int, 3>& kspace_size) const;

  /// @return True if at least one full Brillouin-zone grid was appended.
  bool configure_k_list(
      std::vector<jams::HKLIndex>& k_points,
      std::vector<int>& k_segment_offsets,
      libconfig::Setting& settings,
      const jams::Vec<int, 3>& kspace_size) const;

  void configure_exact_k_list(
      std::vector<jams::HKLIndex>& k_points,
      std::vector<int>& k_segment_offsets,
      libconfig::Setting& hkl_path_settings,
      libconfig::Setting* points_per_segment_settings) const;

private:
  static std::vector<jams::Vec<double, 3>> read_hkl_nodes(libconfig::Setting& settings);

  void append_exact_k_path_segment(
      std::vector<jams::HKLIndex>& k_points,
      std::vector<int>& k_segment_offsets,
      const std::vector<jams::Vec<double, 3>>& hkl_nodes,
      const std::vector<int>& points_per_segment,
      std::size_t& points_offset) const;

  void make_hkl_path(
      const std::vector<jams::Vec<double, 3>>& hkl_nodes,
      const jams::Vec<int, 3>& kspace_size,
      std::vector<jams::HKLIndex>& hkl_path) const;

  Lattice& lattice_;
};

#endif // JAMS_KPOINT_PATH_BUILDER_H
