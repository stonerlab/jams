//
// Created by ioannis charalampidis on 02/08/2022.
//

#include "finite_difference_topological_charge.h"
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/interactions.h>
#include <jams/helpers/montecarlo.h>
#include <jams/containers/interaction_list.h>
#include "jams/helpers/output.h"
#include "jams/core/solver.h"
//#include "jams/interface/config.h"


TopChargeMonitor::TopChargeMonitor(const libconfig::Setting &settings) : Monitor(settings),outfile(jams::output::full_path_filename("top_charge.tsv")) {
  if (!approximately_equal(lattice->a(), {1.0, 0.0, 0.0}, jams::defaults::lattice_tolerance)) {
	throw std::runtime_error("Metadynamics 'topological_charge_finite_difference_monitor' "
							 "requires the 'a' lattice parameter to be (1.0, 0.0, 0.0)");
  }

  if (!approximately_equal(lattice->b(), {0.5, sqrt(3)/2, 0.0}, jams::defaults::lattice_tolerance)) {
	throw std::runtime_error("Metadynamics CV 'topological_charge_finite_diff' "
							 "requires the 'b' lattice parameter to be (0.5, 0.8660254, 0.0)");
  }

//// ------------------------------- ∂ₓS ---------------------------------------
  {
	// first index is interaction vector, second is the +/- sign of the
	// contribution
	std::vector<std::pair<Vec3, int>> dx_interaction_data = {
		{Vec3{1, -1, 0}, +1}, // +S(rᵢ + u₁ - u₂)
		{Vec3{-1, 1, 0}, -1}, // -S(rᵢ - u₁ + u₂)
		{Vec3{0, 1, 0},  +1}, // +S(rᵢ + u₂)
		{Vec3{0, -1, 0}, -1}  // -S(rᵢ - u₂)
	};

	std::vector<InteractionData> interaction_template;
	for (auto &data: dx_interaction_data) {
	  InteractionData J;
	  J.unit_cell_pos_i = 0;
	  J.unit_cell_pos_j = 0;
	  J.type_i = lattice->material_name(
		  lattice->motif_atom(J.unit_cell_pos_i).material_index);
	  J.type_j = lattice->material_name(lattice->motif_atom(J.unit_cell_pos_j).material_index);
	  J.r_ij = ::lattice->fractional_to_cartesian(data.first);
	  J.J_ij[0][0] = data.second;
	  interaction_template.push_back(J);
	}

	jams::InteractionList<Mat3,2> nbrs = neighbour_list_from_interactions(interaction_template);
//	jams::InteractionList<Mat3,2> nbrs = neighbour_list_from_interactions(interaction_templat;e)
	dx_indices_.resize(globals::num_spins);
	dx_values_.resize(globals::num_spins);

	for (auto n = 0; n < nbrs.size(); ++n) {
	  auto i = nbrs[n].first[0];
	  auto j = nbrs[n].first[1];
	  auto sign = nbrs[n].second[0][0];
	  // Divide by 2 for the dx direction
	  dx_indices_[i].push_back(j);
	  dx_values_[i].push_back(sign / 2.0);
	}
  }

  // ------------------------------- ∂ᵧS ---------------------------------------
  {
	// first index is interaction vector, second is the +/- sign of the
	// contribution
	std::vector<std::pair<Vec3, int>> dx_interaction_data = {
		{Vec3{1, -1, 0}, -1}, // -S(rᵢ + u₁ - u₂)
		{Vec3{-1, 1, 0}, +1}, // +S(rᵢ - u₁ + u₂)
		{Vec3{0, 1, 0}, +1}, // +S(rᵢ + u₂)
		{Vec3{0, -1, 0}, -1}  // -S(rᵢ - u₂)
	};

	std::vector<InteractionData> interaction_template;
	for (auto &data: dx_interaction_data) {
	  InteractionData J;
	  J.unit_cell_pos_i = 0;
	  J.unit_cell_pos_j = 0;
	  J.type_i = lattice->material_name(
		  lattice->motif_atom(J.unit_cell_pos_i).material_index);
	  J.type_j = lattice->material_name(
		  lattice->motif_atom(J.unit_cell_pos_j).material_index);
	  J.r_ij = ::lattice->fractional_to_cartesian(data.first);
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
	  auto sign = nbrs[n].second[0][0];

	  // Divide by 2\sqrt{3} for the dy direction
	  dy_indices_[i].push_back(j);
	  dy_values_[i].push_back(sign / (2.0*sqrt(3)));
	}
  }

  max_tolerance_threshold_ = jams::config_optional<double>(settings, "max_threshold",1.0);
  min_tolerance_threshold_ = jams::config_optional<double>(settings,"min_threshold",-1.0);
  tolerance_value_ = jams::config_optional<double>(settings,"tolerance_value",0.05);


  outfile.setf(std::ios::right);
  outfile << tsv_header();

}
bool TopChargeMonitor::is_converged() {
  if (convergence_is_on_ ) {

//	  if (approximately_equal(monitor_top_charge_cache_,max_tolerance_threshold_,tolerance_value_)
//	                 || approximately_equal(monitor_top_charge_cache_,min_tolerance_threshold_,tolerance_value_)) {
      if (monitor_top_charge_cache_ >= max_tolerance_threshold_ || monitor_top_charge_cache_ <= min_tolerance_threshold_ ){
		std::cout << "skyrmion appeared " << monitor_top_charge_cache_ << "\n";
		return true;
	  }
  }
  return false;
}

void TopChargeMonitor::update(Solver *solver) {
  using namespace jams;
  using namespace globals;

  outfile.width(12);
  double topological_charge = 0.0;
  for (auto i = 0; i < globals::num_spins; ++i) {
	topological_charge += local_topological_charge(i);
  }
  monitor_top_charge_cache_ = topological_charge / (4.0 * kPi); //Global Variable I also use it to check the convergence.
  outfile << fmt::sci << solver->iteration()<< "\t";
//  std::cout<<fmt::sci << solver->iteration()<< "\t";
  outfile << fmt::decimal << monitor_top_charge_cache_<< "\t";
//  std::cout<<" ,"<<fmt::decimal << monitor_top_charge_cache_;
  outfile << std::endl;
//  std::cout<<"\n";

}
std::string TopChargeMonitor::tsv_header() {
    using namespace jams;

  std::stringstream ss;
  ss.width(12);

  ss <<fmt::sci << "iteration";
  ss <<fmt::decimal << "TopCharge";

  ss << std::endl;

  return ss.str();

}
double TopChargeMonitor::local_topological_charge(const int i) const {

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

TopChargeMonitor::~TopChargeMonitor() = default;


