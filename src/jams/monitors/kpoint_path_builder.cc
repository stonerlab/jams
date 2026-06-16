//
// Created by Codex on 2026-02-16.
//

#include "jams/monitors/kpoint_path_builder.h"

#include "jams/core/lattice.h"
#include "jams/interface/config.h"
#include "jams/interface/fft.h"

#include <algorithm>
#include <limits>
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
    const jams::Vec<int, 3>& kspace_size) const
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
        const jams::Vec<int, 3> coordinate = {l, m, n};
        const jams::Vec<double, 3> hkl = jams::hadamard_product(coordinate, 1.0 / jams::to_double(kspace_size));
        const jams::Vec<double, 3> xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
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
    const jams::Vec<int, 3>& kspace_size) const
{
  if (!settings.isList())
  {
    throw std::runtime_error("SpectrumBaseMonitor::configure_continuous_kpath failed because settings is not a List");
  }

  std::vector<jams::Vec<double, 3>> hkl_path_nodes(settings.getLength());
  for (auto i = 0; i < settings.getLength(); ++i)
  {
    hkl_path_nodes[i] = jams::read_vec_setting<double, 3>(settings[i], "hkl node");
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
    const jams::Vec<int, 3> start = jams::to_int(jams::hadamard_product(hkl_path_nodes[i - 1], kspace_size));
    const jams::Vec<int, 3> end = jams::to_int(jams::hadamard_product(hkl_path_nodes[i], kspace_size));
    const jams::Vec<int, 3> displacement = jams::absolute(end - start);
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
    const jams::Vec<int, 3>& kspace_size) const
{
  bool full_brillouin_zone_appended = false;

  if (jams::setting_equals_string(settings, "full"))
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
      if (jams::setting_equals_string(settings[n], "full"))
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

std::vector<jams::Vec<double, 3>> KPointPathBuilder::read_hkl_nodes(
    libconfig::Setting& settings)
{
  if (!settings.isList())
  {
    throw std::runtime_error("direct_sum.hkl_path segment must be a list of hkl arrays");
  }

  std::vector<jams::Vec<double, 3>> nodes(settings.getLength());
  for (auto i = 0; i < settings.getLength(); ++i)
  {
    nodes[i] = jams::read_vec_setting<double, 3>(settings[i], "hkl node");
  }
  return nodes;
}

void KPointPathBuilder::configure_exact_k_list(
    std::vector<jams::HKLIndex>& k_points,
    std::vector<int>& k_segment_offsets,
    libconfig::Setting& hkl_path_settings,
    libconfig::Setting* points_per_segment_settings) const
{
  if (jams::setting_equals_string(hkl_path_settings, "full"))
  {
    throw std::runtime_error("direct_sum.hkl_path does not support \"full\"");
  }
  if (!hkl_path_settings.isList())
  {
    throw std::runtime_error("direct_sum.hkl_path must be a list of hkl arrays or path segments");
  }
  if (hkl_path_settings.getLength() <= 0)
  {
    throw std::runtime_error("direct_sum.hkl_path must contain at least one hkl point");
  }

  std::vector<std::vector<jams::Vec<double, 3>>> segments;
  if (hkl_path_settings[0].isArray())
  {
    segments.push_back(read_hkl_nodes(hkl_path_settings));
  }
  else if (hkl_path_settings[0].isList())
  {
    segments.reserve(static_cast<std::size_t>(hkl_path_settings.getLength()));
    for (auto n = 0; n < hkl_path_settings.getLength(); ++n)
    {
      segments.push_back(read_hkl_nodes(hkl_path_settings[n]));
    }
  }
  else
  {
    throw std::runtime_error("direct_sum.hkl_path must contain hkl arrays or path segments");
  }

  std::size_t num_line_segments = 0;
  for (const auto& segment : segments)
  {
    if (segment.empty())
    {
      throw std::runtime_error("direct_sum.hkl_path segment must contain at least one hkl point");
    }
    if (segment.size() > 1)
    {
      num_line_segments += segment.size() - 1;
    }
  }

  std::vector<int> points_per_segment;
  if (num_line_segments > 0)
  {
    if (!points_per_segment_settings)
    {
      throw std::runtime_error("direct_sum.points_per_segment is required for hkl paths with line segments");
    }

    if (jams::is_integer_setting(*points_per_segment_settings))
    {
      const int count = jams::read_integer_setting(*points_per_segment_settings, "direct_sum.points_per_segment");
      if (count < 2)
      {
        throw std::runtime_error("direct_sum.points_per_segment must be at least 2 for a line segment");
      }
      points_per_segment.assign(num_line_segments, count);
    }
    else if (jams::is_sequence_setting(*points_per_segment_settings))
    {
      if (points_per_segment_settings->getLength() != static_cast<int>(num_line_segments))
      {
        throw std::runtime_error("direct_sum.points_per_segment list length must match the number of hkl path line segments");
      }
      points_per_segment.reserve(num_line_segments);
      for (auto i = 0; i < points_per_segment_settings->getLength(); ++i)
      {
        const int count = jams::read_integer_setting((*points_per_segment_settings)[i], "direct_sum.points_per_segment");
        if (count < 2)
        {
          throw std::runtime_error("direct_sum.points_per_segment entries must be at least 2");
        }
        points_per_segment.push_back(count);
      }
    }
    else
    {
      throw std::runtime_error("direct_sum.points_per_segment must be an integer or list of integers");
    }
  }

  std::size_t points_offset = 0;
  for (const auto& segment : segments)
  {
    append_exact_k_path_segment(
        k_points,
        k_segment_offsets,
        segment,
        points_per_segment,
        points_offset);
  }
  if (points_offset != points_per_segment.size())
  {
    throw std::runtime_error("direct_sum.points_per_segment was not fully consumed");
  }
}

void KPointPathBuilder::append_exact_k_path_segment(
    std::vector<jams::HKLIndex>& k_points,
    std::vector<int>& k_segment_offsets,
    const std::vector<jams::Vec<double, 3>>& hkl_nodes,
    const std::vector<int>& points_per_segment,
    std::size_t& points_offset) const
{
  if (hkl_nodes.empty())
  {
    throw std::runtime_error("direct_sum.hkl_path segment must contain at least one hkl point");
  }

  if (k_segment_offsets.empty())
  {
    k_segment_offsets.push_back(0);
  }

  const std::size_t initial_size = k_points.size();
  const auto push_unique = [&](const jams::Vec<double, 3>& hkl)
  {
    const jams::Vec<double, 3> xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
    const FFTWHermitianIndex<3> no_fft_index{{0, 0, 0}, false};
    const jams::HKLIndex point{hkl, xyz, no_fft_index, false};
    if (k_points.size() > initial_size && k_points.back() == point)
    {
      return;
    }
    k_points.push_back(point);
  };

  if (hkl_nodes.size() == 1)
  {
    push_unique(hkl_nodes.front());
  }
  else
  {
    for (std::size_t n = 0; n + 1 < hkl_nodes.size(); ++n)
    {
      if (hkl_nodes[n] == hkl_nodes[n + 1])
      {
        throw std::runtime_error("Two consecutive direct_sum hkl_path nodes cannot be the same");
      }
      if (points_offset >= points_per_segment.size())
      {
        throw std::runtime_error("direct_sum.points_per_segment has too few entries");
      }
      const int count = points_per_segment[points_offset++];
      for (int i = 0; i < count; ++i)
      {
        const double alpha = static_cast<double>(i) / static_cast<double>(count - 1);
        const auto hkl = (1.0 - alpha) * hkl_nodes[n] + alpha * hkl_nodes[n + 1];
        push_unique(hkl);
      }
    }
  }

  const std::size_t added_count = k_points.size() - initial_size;
  if (added_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error("direct_sum.hkl_path segment has too many points");
  }
  k_segment_offsets.push_back(k_segment_offsets.back() + static_cast<int>(added_count));
}

void KPointPathBuilder::make_hkl_path(
    const std::vector<jams::Vec<double, 3>>& hkl_nodes,
    const jams::Vec<int, 3>& kspace_size,
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
    jams::Vec<int, 3> start = jams::to_int(jams::hadamard_product(hkl_nodes[n], kspace_size));
    jams::Vec<int, 3> end = jams::to_int(jams::hadamard_product(hkl_nodes[n + 1], kspace_size));
    jams::Vec<int, 3> displacement = jams::absolute(end - start);

    jams::Vec<int, 3> step = {
        (end[0] > start[0]) ? 1 : ((end[0] < start[0]) ? -1 : 0),
        (end[1] > start[1]) ? 1 : ((end[1] < start[1]) ? -1 : 0),
        (end[2] > start[2]) ? 1 : ((end[2] < start[2]) ? -1 : 0)};

    if (displacement[0] >= displacement[1] && displacement[0] >= displacement[2])
    {
      int p1 = 2 * displacement[1] - displacement[0];
      int p2 = 2 * displacement[2] - displacement[0];
      while (start[0] != end[0])
      {
        const jams::Vec<double, 3> hkl = jams::hadamard_product(start, 1.0 / jams::to_double(kspace_size));
        const jams::Vec<double, 3> xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
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
        const jams::Vec<double, 3> hkl = jams::hadamard_product(start, 1.0 / jams::to_double(kspace_size));
        const jams::Vec<double, 3> xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
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
        const jams::Vec<double, 3> hkl = jams::hadamard_product(start, 1.0 / jams::to_double(kspace_size));
        const jams::Vec<double, 3> xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
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

    const jams::Vec<double, 3> hkl = jams::hadamard_product(end, 1.0 / jams::to_double(kspace_size));
    const jams::Vec<double, 3> xyz = lattice_.get_unitcell().inv_fractional_to_cartesian(hkl);
    push_unique(jams::HKLIndex{hkl, xyz, fftw_r2c_index(end, kspace_size)});
  }
}
