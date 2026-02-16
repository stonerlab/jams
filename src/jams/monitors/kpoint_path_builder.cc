//
// Created by Codex on 2026-02-16.
//

#include "jams/monitors/kpoint_path_builder.h"

#include "jams/core/lattice.h"
#include "jams/interface/fft.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

KPointPathBuilder::KPointPathBuilder(Lattice& lattice)
    : lattice_(lattice)
{
}

void KPointPathBuilder::append_full_k_grid(
    std::vector<jams::HKLIndex>& k_points,
    std::vector<int>& k_segment_offsets,
    const Vec3i& kspace_size) const
{
  const std::size_t initial_size = k_points.size();
  const std::size_t added_count_estimate = static_cast<std::size_t>(jams::product(kspace_size));
  k_points.reserve(initial_size + added_count_estimate);

  for (auto l = 0; l < kspace_size[0]; ++l)
  {
    for (auto m = 0; m < kspace_size[1]; ++m)
    {
      for (auto n = 0; n < kspace_size[2]; ++n)
      {
        const Vec3i coordinate = {l, m, n};
        const Vec3 hkl = jams::hadamard_product(coordinate, 1.0 / jams::to_double(kspace_size));
        const Vec3 xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
        k_points.push_back(jams::HKLIndex{hkl, xyz, fftw_r2c_index(coordinate, kspace_size)});
      }
    }
  }

  if (k_segment_offsets.empty())
  {
    k_segment_offsets.push_back(0);
  }
  const std::size_t added_count = k_points.size() - initial_size;
  k_segment_offsets.push_back(k_segment_offsets.back() + static_cast<int>(added_count));
}

void KPointPathBuilder::append_k_path_segment(
    std::vector<jams::HKLIndex>& k_points,
    std::vector<int>& k_segment_offsets,
    libconfig::Setting& settings,
    const Vec3i& kspace_size) const
{
  if (!settings.isList())
  {
    throw std::runtime_error("SpectrumBaseMonitor::configure_continuous_kpath failed because settings is not a List");
  }

  std::vector<Vec3> hkl_path_nodes(settings.getLength());
  for (auto i = 0; i < settings.getLength(); ++i)
  {
    if (!settings[i].isArray())
    {
      throw std::runtime_error("SpectrumBaseMonitor::configure_continuous_kpath failed hkl node is not an Array");
    }

    hkl_path_nodes[i] = Vec3{settings[i][0], settings[i][1], settings[i][2]};
  }

  for (auto i = 1; i < hkl_path_nodes.size(); ++i)
  {
    if (hkl_path_nodes[i] == hkl_path_nodes[i - 1])
    {
      throw std::runtime_error("Two consecutive hkl_nodes cannot be the same");
    }
  }

  std::size_t expected_new_points = 0;
  for (std::size_t i = 1; i < hkl_path_nodes.size(); ++i)
  {
    const Vec3i start = jams::to_int(jams::hadamard_product(hkl_path_nodes[i - 1], kspace_size));
    const Vec3i end = jams::to_int(jams::hadamard_product(hkl_path_nodes[i], kspace_size));
    const Vec3i displacement = jams::absolute(end - start);
    expected_new_points += static_cast<std::size_t>(
        std::max({displacement[0], displacement[1], displacement[2]})) + 1;
  }
  if (hkl_path_nodes.size() > 2)
  {
    expected_new_points -= static_cast<std::size_t>(hkl_path_nodes.size() - 2);
  }

  const std::size_t initial_size = k_points.size();
  k_points.reserve(initial_size + expected_new_points);
  make_hkl_path(hkl_path_nodes, kspace_size, k_points);

  if (k_segment_offsets.empty())
  {
    k_segment_offsets.push_back(0);
  }
  const std::size_t added_count = k_points.size() - initial_size;
  k_segment_offsets.push_back(k_segment_offsets.back() + static_cast<int>(added_count));
}

bool KPointPathBuilder::configure_k_list(
    std::vector<jams::HKLIndex>& k_points,
    std::vector<int>& k_segment_offsets,
    libconfig::Setting& settings,
    const Vec3i& kspace_size) const
{
  bool full_brillouin_zone_appended = false;

  if (settings.isString() && std::string(settings.c_str()) == "full")
  {
    append_full_k_grid(k_points, k_segment_offsets, kspace_size);
    return true;
  }

  if (settings[0].isArray())
  {
    append_k_path_segment(k_points, k_segment_offsets, settings, kspace_size);
    return false;
  }

  if (settings[0].isList())
  {
    for (auto n = 0; n < settings.getLength(); ++n)
    {
      if (settings[n].isArray())
      {
        append_k_path_segment(k_points, k_segment_offsets, settings[n], kspace_size);
        continue;
      }
      if (settings[n].isString() && std::string(settings[n].c_str()) == "full")
      {
        append_full_k_grid(k_points, k_segment_offsets, kspace_size);
        full_brillouin_zone_appended = true;
        continue;
      }
      throw std::runtime_error("SpectrumBaseMonitor::configure_k_list failed because a nodes is not an Array or String");
    }
    return full_brillouin_zone_appended;
  }

  throw std::runtime_error("SpectrumBaseMonitor::configure_k_list failed because settings is not an Array, List or String");
}

void KPointPathBuilder::make_hkl_path(
    const std::vector<Vec3>& hkl_nodes,
    const Vec3i& kspace_size,
    std::vector<jams::HKLIndex>& hkl_path) const
{
  const std::size_t initial_size = hkl_path.size();
  const auto push_unique = [&](const jams::HKLIndex& point)
  {
    if (hkl_path.size() > initial_size && hkl_path.back() == point)
    {
      return;
    }
    hkl_path.push_back(point);
  };

  for (auto n = 0; n < static_cast<int>(hkl_nodes.size()) - 1; ++n)
  {
    Vec3i start = jams::to_int(jams::hadamard_product(hkl_nodes[n], kspace_size));
    Vec3i end = jams::to_int(jams::hadamard_product(hkl_nodes[n + 1], kspace_size));
    Vec3i displacement = jams::absolute(end - start);

    Vec3i step = {
        (end[0] > start[0]) ? 1 : ((end[0] < start[0]) ? -1 : 0),
        (end[1] > start[1]) ? 1 : ((end[1] < start[1]) ? -1 : 0),
        (end[2] > start[2]) ? 1 : ((end[2] < start[2]) ? -1 : 0)};

    if (displacement[0] >= displacement[1] && displacement[0] >= displacement[2])
    {
      int p1 = 2 * displacement[1] - displacement[0];
      int p2 = 2 * displacement[2] - displacement[0];
      while (start[0] != end[0])
      {
        const Vec3 hkl = jams::hadamard_product(start, 1.0 / jams::to_double(kspace_size));
        const Vec3 xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
        push_unique(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[0] += step[0];
        if (p1 >= 0)
        {
          start[1] += step[1];
          p1 -= 2 * displacement[0];
        }
        if (p2 >= 0)
        {
          start[2] += step[2];
          p2 -= 2 * displacement[0];
        }
        p1 += 2 * displacement[1];
        p2 += 2 * displacement[2];
      }
    }
    else if (displacement[1] >= displacement[0] && displacement[1] >= displacement[2])
    {
      int p1 = 2 * displacement[0] - displacement[1];
      int p2 = 2 * displacement[2] - displacement[1];
      while (start[1] != end[1])
      {
        const Vec3 hkl = jams::hadamard_product(start, 1.0 / jams::to_double(kspace_size));
        const Vec3 xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
        push_unique(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[1] += step[1];
        if (p1 >= 0)
        {
          start[0] += step[0];
          p1 -= 2 * displacement[1];
        }
        if (p2 >= 0)
        {
          start[2] += step[2];
          p2 -= 2 * displacement[1];
        }
        p1 += 2 * displacement[0];
        p2 += 2 * displacement[2];
      }
    }
    else
    {
      int p1 = 2 * displacement[0] - displacement[2];
      int p2 = 2 * displacement[1] - displacement[2];
      while (start[2] != end[2])
      {
        const Vec3 hkl = jams::hadamard_product(start, 1.0 / jams::to_double(kspace_size));
        const Vec3 xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
        push_unique(jams::HKLIndex{hkl, xyz, fftw_r2c_index(start, kspace_size)});

        start[2] += step[2];
        if (p1 >= 0)
        {
          start[1] += step[1];
          p1 -= 2 * displacement[2];
        }
        if (p2 >= 0)
        {
          start[0] += step[0];
          p2 -= 2 * displacement[2];
        }
        p1 += 2 * displacement[1];
        p2 += 2 * displacement[0];
      }
    }

    const Vec3 hkl = jams::hadamard_product(end, 1.0 / jams::to_double(kspace_size));
    const Vec3 xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
    push_unique(jams::HKLIndex{hkl, xyz, fftw_r2c_index(end, kspace_size)});
  }
}
