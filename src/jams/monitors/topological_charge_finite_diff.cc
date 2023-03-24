//
// Created by ioannis charalampidis on 02/08/2022.
//

#include <jams/monitors/topological_charge_finite_diff.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/interactions.h>
#include <jams/helpers/montecarlo.h>
#include <jams/containers/interaction_list.h>
#include "jams/helpers/output.h"
#include "jams/core/solver.h"


TopologicalFiniteDiffChargeMonitor::TopologicalFiniteDiffChargeMonitor(const libconfig::Setting &settings) : Monitor(
  settings), outfile(jams::output::full_path_filename("top_charge.tsv")) {

  // calculate topological charge per material or per unit cell position
  auto grouping_str = jams::config_optional<std::string>(settings, "grouping", "positions");

  if (lowercase(grouping_str) == "materials") {
    grouping_ = Grouping::MATERIALS;
  } else if (lowercase(grouping_str) == "positions") {
    grouping_ = Grouping::POSITIONS;
  } else {
    throw std::runtime_error("unknown topological charge grouping: " + grouping_str);
  }

  if (grouping_ == Grouping::MATERIALS) {
    // TODO: We can't reliably use globals::lattice->num_materials() here because we might
    // have specified some materials in the cfg which are never actually used in the lattice.
    // Need to work out how best to fix this. Do we enforce that all materials should be used,
    // or should we make sure num_materials() only returns the number of materials used in the
    // lattice.
    std::vector<std::vector<int>> material_index_groups(globals::lattice->num_materials());
    for (auto i = 0; i < globals::num_spins; ++i) {
      auto type = globals::lattice->atom_material_id(i);
      material_index_groups[type].push_back(i);
    }

    group_spin_indicies_.resize(material_index_groups.size());
    for (auto n = 0; n < material_index_groups.size(); ++n) {
      group_spin_indicies_[n] = jams::MultiArray<int,1>(material_index_groups[n].begin(), material_index_groups[n].end());
    }
  } else if (grouping_ == Grouping::POSITIONS) {
    std::vector<std::vector<int>> position_index_groups(globals::lattice->num_motif_atoms());
    for (auto i = 0; i < globals::num_spins; ++i) {
      auto position = globals::lattice->atom_motif_position(i);
      position_index_groups[position].push_back(i);
    }

    group_spin_indicies_.resize(position_index_groups.size());
    for (auto n = 0; n < position_index_groups.size(); ++n) {
      group_spin_indicies_[n] = jams::MultiArray<int,1>(position_index_groups[n].begin(), position_index_groups[n].end());
    }
  }

  // true if a and b are equal to the lattice a and b vectors.
  auto lattice_equal = [&](Vec3 a, Vec3 b) {
      return approximately_equal(globals::lattice->a(), a, jams::defaults::lattice_tolerance)
             && approximately_equal(globals::lattice->b(), b, jams::defaults::lattice_tolerance);
  };

  // Detect if we have a square or hexagonal lattice in the plane (all other
  // lattices are unsupported)
  enum class LatticeShape {
      Unsupported, Square, HexagonalAcute, HexagonalObtuse
  };

  LatticeShape lattice_shape = LatticeShape::Unsupported;
  if (lattice_equal({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0})) {
    lattice_shape = LatticeShape::Square;
  } else if (lattice_equal({1.0, 0.0, 0.0}, {0.5, sqrt(3) / 2, 0.0})) {
    lattice_shape = LatticeShape::HexagonalAcute;
  } else if (lattice_equal({1.0, 0.0, 0.0}, {-0.5, sqrt(3) / 2, 0.0})) {
    lattice_shape = LatticeShape::HexagonalObtuse;
  } else {
    throw std::runtime_error("Monitor 'topological_charge_finite_diff' "
                             "requires the lattice to be either square or triangular.");
  }

  max_tolerance_threshold_ = jams::config_optional<double>(settings, "max_threshold", 1.0);
  min_tolerance_threshold_ = jams::config_optional<double>(settings, "min_threshold", -1.0);

// ------------------------------- ∂S/∂x ---------------------------------------
  {

    std::vector<std::pair<Vec3, double>> dx_interaction_data;

    if (lattice_shape == LatticeShape::Square) {
      dx_interaction_data = {
        {Vec3{1, 1, 0},   1.0/16.0},
        {Vec3{1, -1, 0},  1.0/16.0},
        {Vec3{1, 0, 0},   3.0/8.0},
        {Vec3{-1, 0, 0},  -3.0/8.0},
        {Vec3{-1, 1, 0},  -1.0/16.0},
        {Vec3{-1, -1, 0}, -1.0/16.0},
      };
    }
    if (lattice_shape == LatticeShape::HexagonalAcute) {
      dx_interaction_data = {
        {Vec3{1, -1, 0}, 1.0/6.0},
        {Vec3{-1, 1, 0}, -1.0/6.0},
        {Vec3{0, 1, 0},  1.0/6.0},
        {Vec3{0, -1, 0}, -1.0/6.0},
        {Vec3{1, 0, 0},  1.0/3.0},
        {Vec3{-1, 0, 0}, -1.0/3.0}
      };
    }

    if (lattice_shape == LatticeShape::HexagonalObtuse) {
      dx_interaction_data = {
        {Vec3{0, 1, 0},   -1.0/6.0},
        {Vec3{1, 1, 0},   1.0/6.0},
        {Vec3{1, 0, 0},   1.0/3.0},
        {Vec3{0, -1, 0},  1.0/6.0},
        {Vec3{-1, -1, 0}, -1.0/6.0},
        {Vec3{-1, 0, 0},  -1.0/3.0},
      };
    }

    std::vector<InteractionData> interaction_template;
    for (auto &data: dx_interaction_data) {
      // To work with the neighbour_list_from_interactions() mechanics we need
      // to specify the unit cell positions. We assume that each plane only
      // contains one motif position.
      for (auto m = 0; m < globals::lattice->num_motif_atoms(); ++m) {
        InteractionData J;
        J.unit_cell_pos_i = m;
        J.unit_cell_pos_j = m;
        J.type_i = globals::lattice->material_name(
            globals::lattice->motif_atom(J.unit_cell_pos_i).material_index);
        J.type_j = globals::lattice->material_name(globals::lattice->motif_atom(J.unit_cell_pos_j).material_index);
        J.r_ij = ::globals::lattice->fractional_to_cartesian(data.first);
        J.J_ij[0][0] = data.second;
        interaction_template.push_back(J);
      }
    }

    jams::InteractionList<Mat3, 2> nbrs = neighbour_list_from_interactions(interaction_template);
    dx_indices_.resize(globals::num_spins);
    dx_values_.resize(globals::num_spins);

    for (auto n = 0; n < nbrs.size(); ++n) {
      auto i = nbrs[n].first[0];
      auto j = nbrs[n].first[1];
      auto weight = nbrs[n].second[0][0];
      dx_indices_[i].push_back(j);
      dx_values_[i].push_back(weight);
    }
  }

  // ------------------------------- ∂S/∂y -------------------------------------
  {
    // first index is interaction vector, second is the +/- sign of the
    // contribution
    std::vector<std::pair<Vec3, double>> dy_interaction_data;

    if (lattice_shape == LatticeShape::Square) {
      dy_interaction_data = {
        {Vec3{1, 1, 0},   1.0/16.0},
        {Vec3{-1, 1, 0},  1.0/16.0},
        {Vec3{0, 1, 0},   3.0/8.0},
        {Vec3{0, -1, 0},  -3.0/8.0},
        {Vec3{1, -1, 0},  -1.0/16.0},
        {Vec3{-1, -1, 0}, -1.0/16.0},
      };
    }
    if (lattice_shape == LatticeShape::HexagonalAcute) {
      dy_interaction_data = {
        {Vec3{1, -1, 0}, -sqrt(3.0)/6.0},
        {Vec3{-1, 1, 0}, sqrt(3.0)/6.0},
        {Vec3{0, 1, 0},  sqrt(3.0)/6.0},
        {Vec3{0, -1, 0}, -sqrt(3.0)/6.0},
      };
    }
    if (lattice_shape == LatticeShape::HexagonalObtuse) {
      dy_interaction_data = {
        {Vec3{0, 1, 0},   sqrt(3.0)/6.0},
        {Vec3{1, 1, 0},   sqrt(3.0)/6.0},
        {Vec3{0, -1, 0},  -sqrt(3.0)/6.0},
        {Vec3{-1, -1, 0}, -sqrt(3.0)/6.0},
      };
    }
    std::vector<InteractionData> interaction_template;
    for (auto &data: dy_interaction_data) {
      for (auto m = 0; m < globals::lattice->num_motif_atoms(); ++m) {
        InteractionData J;
        J.unit_cell_pos_i = m;
        J.unit_cell_pos_j = m;
        J.type_i = globals::lattice->material_name(
            globals::lattice->motif_atom(J.unit_cell_pos_i).material_index);
        J.type_j = globals::lattice->material_name(
            globals::lattice->motif_atom(J.unit_cell_pos_j).material_index);
        J.r_ij = ::globals::lattice->fractional_to_cartesian(data.first);
        J.J_ij[0][0] = data.second;
        interaction_template.push_back(J);
      }
    }
    jams::InteractionList<Mat3, 2> nbrs = neighbour_list_from_interactions(
      interaction_template);
    dy_indices_.resize(globals::num_spins);
    dy_values_.resize(globals::num_spins);

    for (auto n = 0; n < nbrs.size(); ++n) {
      auto i = nbrs[n].first[0];
      auto j = nbrs[n].first[1];
      auto weight = nbrs[n].second[0][0];

      dy_indices_[i].push_back(j);
      dy_values_[i].push_back(weight);
    }
  }

  if (grouping_ == Grouping::MATERIALS) {
    monitor_top_charge_cache_.resize(globals::lattice->num_materials());
  } else if (grouping_ == Grouping::POSITIONS) {
    monitor_top_charge_cache_.resize(globals::lattice->num_motif_atoms());
  }

  outfile.setf(std::ios::right);
  outfile << tsv_header();

}

Monitor::ConvergenceStatus TopologicalFiniteDiffChargeMonitor::convergence_status() {
  if (convergence_status_ == Monitor::ConvergenceStatus::kDisabled) {
    return convergence_status_;
  }

  convergence_status_ = Monitor::ConvergenceStatus::kNotConverged;

  bool layer_convergence = false;
  for (auto n = 0; n < monitor_top_charge_cache_.size(); ++n) {
    layer_convergence = layer_convergence &&
        (greater_than_approx_equal(monitor_top_charge_cache_[n], max_tolerance_threshold_, 1e-5)
                        || less_than_approx_equal(monitor_top_charge_cache_[n], min_tolerance_threshold_, 1e-5));
  }
  if (layer_convergence) {
    convergence_status_ = Monitor::ConvergenceStatus::kConverged;
  }

  return convergence_status_;
}

void TopologicalFiniteDiffChargeMonitor::update(Solver &solver) {
  outfile << jams::fmt::sci << solver.time();
  outfile.width(12);
  outfile << jams::fmt::sci << solver.iteration() << "\t";

  for (auto n = 0; n < group_spin_indicies_.size(); ++n) {
    double topological_charge = topological_charge_from_indices(group_spin_indicies_[n]);
    outfile << jams::fmt::decimal << topological_charge / (4.0 * kPi);
  }

  outfile << std::endl;
}

std::string TopologicalFiniteDiffChargeMonitor::tsv_header() {
  using namespace jams;

  std::stringstream ss;
  ss.width(12);

  ss << fmt::sci << "time";
  ss << fmt::sci << "iteration";

  if (grouping_ == Grouping::MATERIALS) {
    for (auto i = 0; i < globals::lattice->num_materials(); ++i) {
      auto name = globals::lattice->material_name(i);
      ss << fmt::decimal << name;
    }
  } else if (grouping_ == Grouping::POSITIONS) {
    for (auto i = 0; i < globals::lattice->num_motif_atoms(); ++i) {
      auto material_name = globals::lattice->material_name(
          globals::lattice->motif_atom(i).material_index);
      ss << fmt::sci << std::to_string(i+1) + "_" + material_name;
    }
  }

  ss << std::endl;


  return ss.str();

}

double TopologicalFiniteDiffChargeMonitor::local_topological_charge(const int i) const {

  Vec3 ds_x = {0.0, 0.0, 0.0};
  for (auto n = 0; n < dx_indices_[i].size(); ++n) {
    ds_x += dx_values_[i][n] * jams::montecarlo::get_spin(dx_indices_[i][n]);
  }

  Vec3 ds_y = {0.0, 0.0, 0.0};
  for (auto n = 0; n < dy_indices_[i].size(); ++n) {
    ds_y += dy_values_[i][n] * jams::montecarlo::get_spin(dy_indices_[i][n]);
  }

  Vec3 s_i = jams::montecarlo::get_spin(i);

  return dot(s_i, cross(ds_x, ds_y));
}


double TopologicalFiniteDiffChargeMonitor::topological_charge_from_indices(
    const jams::MultiArray<int, 1> &indices) const {
  double sum = 0.0;
  double c = 0.0;
  for (auto i = 0; i < indices.size(); ++i) {
    const auto idx = indices(i);
    // Use a Kahan sum incase we're adding some very small charges.
    double y = local_topological_charge(idx) - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

