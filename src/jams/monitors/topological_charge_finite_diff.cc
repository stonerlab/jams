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


TopologicalFiniteDiffChargeMonitor::TopologicalFiniteDiffChargeMonitor(const libconfig::Setting &settings) : Monitor(settings), outfile(jams::output::full_path_filename("top_charge.tsv")) {

  if (settings.exists("material")) {
    std::string material = jams::config_optional<std::string>(settings, material, "all");

    if (!globals::lattice->material_exists(material)) {
      throw std::runtime_error("Invalid material specified in topological charge collective variable.");
    }
    selected_material_id_ = globals::lattice->material_id(material);

    num_selected_layers_ = 0;
    for (auto i = 0; i < globals::lattice->num_motif_atoms(); ++i) {
      // iterate over layers
      if (globals::lattice->motif_atom(i).material_index == selected_material_id_) {
        num_selected_layers_ += 1;
      }
    }
  } else {
    selected_material_id_ = -1;
    num_selected_layers_ = globals::lattice->num_motif_atoms();
  }
  // true if a and b are equal to the lattice a and b vectors.
  auto lattice_equal = [&](Vec3 a, Vec3 b) {
      return approximately_equal(globals::lattice->a(), a, jams::defaults::lattice_tolerance)
             && approximately_equal(globals::lattice->b(), b, jams::defaults::lattice_tolerance);
  };

  // Detect if we have a square or hexagonal lattice in the plane (all other
  // lattices are unsupported)
  enum class LatticeShape {Unsupported, Square, Hexagonal};

  LatticeShape lattice_shape = LatticeShape::Unsupported;
  if (lattice_equal({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0})) {
    lattice_shape = LatticeShape::Square;
  } else if (lattice_equal({1.0, 0.0, 0.0}, {0.5, sqrt(3)/2, 0.0})) {
    lattice_shape = LatticeShape::Hexagonal;
  } else {
    throw std::runtime_error("Monitor 'topological_charge_finite_diff' "
                             "requires the lattice to be either square or triangular.");
  }

  max_tolerance_threshold_ = jams::config_optional<double>(settings, "max_threshold",1.0);
  min_tolerance_threshold_ = jams::config_optional<double>(settings,"min_threshold",-1.0);

// ------------------------------- ∂S/∂x ---------------------------------------
  {

    std::vector<std::pair<Vec3, double>> dx_interaction_data;

    if (lattice_shape == LatticeShape::Square) {
      dx_interaction_data = {
          {Vec3{ 1, 1, 0}, 1.0/16.0},
          {Vec3{ 1,-1, 0}, 1.0/16.0},
          {Vec3{ 1, 0, 0}, 3.0/8.0},
          {Vec3{-1, 0, 0},-3.0/8.0},
          {Vec3{-1, 1, 0},-1.0/16.0},
          {Vec3{-1,-1, 0},-1.0/16.0},
      };
    }
    if (lattice_shape == LatticeShape::Hexagonal) {
      dx_interaction_data = {
          {Vec3{ 1,-1, 0}, 1.0/6.0},
          {Vec3{-1, 1, 0},-1.0/6.0},
          {Vec3{ 0, 1, 0}, 1.0/6.0},
          {Vec3{ 0,-1, 0},-1.0/6.0},
          {Vec3{ 1, 0, 0}, 1.0/3.0},
          {Vec3{-1, 0, 0},-1.0/3.0}
      };
    }

	std::vector<InteractionData> interaction_template;
	for (auto &data: dx_interaction_data) {
	  InteractionData J;
	  J.unit_cell_pos_i = 0;
	  J.unit_cell_pos_j = 0;
	  J.type_i = globals::lattice->material_name(
		  globals::lattice->motif_atom(J.unit_cell_pos_i).material_index);
	  J.type_j = globals::lattice->material_name(globals::lattice->motif_atom(J.unit_cell_pos_j).material_index);
	  J.r_ij = ::globals::lattice->fractional_to_cartesian(data.first);
	  J.J_ij[0][0] = data.second;
	  interaction_template.push_back(J);
	}

	jams::InteractionList<Mat3,2> nbrs = neighbour_list_from_interactions(interaction_template);
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
          {Vec3{ 1, 1, 0}, 1.0/16.0},
          {Vec3{-1, 1, 0}, 1.0/16.0},
          {Vec3{ 0, 1, 0}, 3.0/8.0},
          {Vec3{ 0,-1, 0},-3.0/8.0},
          {Vec3{ 1,-1, 0},-1.0/16.0},
          {Vec3{-1,-1, 0},-1.0/16.0},
      };
    }
    if (lattice_shape == LatticeShape::Hexagonal) {
      dy_interaction_data = {
          {Vec3{ 1,-1, 0},-sqrt(3.0)/6.0},
          {Vec3{-1, 1, 0}, sqrt(3.0)/6.0},
          {Vec3{ 0, 1, 0}, sqrt(3.0)/6.0},
          {Vec3{ 0,-1, 0},-sqrt(3.0)/6.0},
      };
    }

	std::vector<InteractionData> interaction_template;
	for (auto &data: dy_interaction_data) {
	  InteractionData J;
	  J.unit_cell_pos_i = 0;
	  J.unit_cell_pos_j = 0;
	  J.type_i = globals::lattice->material_name(
		  globals::lattice->motif_atom(J.unit_cell_pos_i).material_index);
	  J.type_j = globals::lattice->material_name(
		  globals::lattice->motif_atom(J.unit_cell_pos_j).material_index);
	  J.r_ij = ::globals::lattice->fractional_to_cartesian(data.first);
	  J.J_ij[0][0] = data.second;
	  interaction_template.push_back(J);
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


  outfile.setf(std::ios::right);
  outfile << tsv_header();

}
Monitor::ConvergenceStatus TopologicalFiniteDiffChargeMonitor::convergence_status() {
  if (convergence_status_ == Monitor::ConvergenceStatus::kDisabled) {
    return convergence_status_;
  }

  convergence_status_ = Monitor::ConvergenceStatus::kNotConverged;
  if (greater_than_approx_equal(monitor_top_charge_cache_, max_tolerance_threshold_, 1e-5)
      || less_than_approx_equal(monitor_top_charge_cache_, min_tolerance_threshold_, 1e-5)) {
    convergence_status_ =  Monitor::ConvergenceStatus::kConverged;
  }

  return convergence_status_;
}

void TopologicalFiniteDiffChargeMonitor::update(Solver& solver) {
  double sum = 0.0;
  double c = 0.0;

  for (auto i = 0; i < globals::num_spins; ++i) {
    if (selected_material_id_==-1 || globals::lattice->atom_material_id(i) == selected_material_id_) {
      double y = local_topological_charge(i) - c;
      double t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
  }

  monitor_top_charge_cache_ = sum / (4.0 * kPi * num_selected_layers_);

  outfile.width(12);
  outfile << jams::fmt::sci << solver.iteration()<< "\t";
  outfile << jams::fmt::decimal << monitor_top_charge_cache_ << "\t";
  outfile << std::endl;
}
std::string TopologicalFiniteDiffChargeMonitor::tsv_header() {
    using namespace jams;

  std::stringstream ss;
  ss.width(12);

  ss <<fmt::sci << "time";
  ss <<fmt::decimal << "topological_charge";

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

