//
// Created by ioannis charalampidis on 17/08/2022.
//

#include "jams/monitors/topological_charge_geometrical_def.h"
#include <jams/core/lattice.h>
#include <jams/core/interactions.h>
#include <jams/containers/interaction_list.h>
#include "jams/helpers/output.h"
#include "jams/core/solver.h"
#include <jams/core/globals.h>
#include <jams/lattice/interaction_neartree.h>
#include <unordered_set>
#include <vector>
#include <set>
#include <jams/helpers/montecarlo.h>




TopologicalGeometricalDefMonitor::TopologicalGeometricalDefMonitor(const libconfig::Setting &settings) : Monitor(
	settings), outfile(jams::output::full_path_filename("top_geom_def_charge.tsv")) {
  if (settings.exists("radius_cutoff")) {
	radius_cutoff_ = settings["radius_cutoff"];
  }
  std::cout << "  radius_cutoff: " << radius_cutoff_ << std::endl;

  // Calculate the number of layers along the z direction
  auto comp = [](double a, double b){ return definately_less_than(a, b, jams::defaults::lattice_tolerance); };
  std::set<double, decltype(comp)> z_indices(comp);
  for (const auto& r : globals::lattice->atom_cartesian_positions()) {
	z_indices.insert(r[2]);
  }
  recip_num_layers_ = 1.0 / z_indices.size();

  std::cout << "  num_layers: " << z_indices.size() << std::endl;

  calculate_elementary_triangles();

  std::cout << "  num_triangles: " << triangle_indices_.size() << std::endl;

  max_tolerance_threshold_ = jams::config_optional<double>(settings, "max_threshold",1.0);
  min_tolerance_threshold_ = jams::config_optional<double>(settings,"min_threshold",-1.0);

  outfile.setf(std::ios::right);
  outfile << tsv_header();

}
void TopologicalGeometricalDefMonitor::update(Solver& solver) {
  monitor_top_charge_cache_ = total_topological_charge();

  outfile.width(12);
  outfile << jams::fmt::sci << solver.iteration()<< "\t";
  outfile << jams::fmt::decimal << monitor_top_charge_cache_ << "\t";
  outfile << std::endl;
}

Monitor::ConvergenceStatus TopologicalGeometricalDefMonitor::convergence_status() {
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

// Private Functions

void TopologicalGeometricalDefMonitor::calculate_elementary_triangles() {

  // Generate a near tree of nearest neighbours from which we can calculate the triangle indices
  jams::InteractionNearTree neartree(globals::lattice->get_supercell().a(), globals::lattice->get_supercell().b(), globals::lattice->get_supercell().c(), globals::lattice->periodic_boundaries(), radius_cutoff_, jams::defaults::lattice_tolerance);
  neartree.insert_sites(globals::lattice->atom_cartesian_positions());

  // Use a set to ensure only a single version of equivalent triangles is stored.
  // This uses the structs defined in the class to compare ijk triplets to
  // determine which are duplicates. A duplicate is when a triplet which is being
  // inserted is either the same as an existing triplet or is a cyclic permutation
  // with the same handedness (clockwise or anticlockwise). For the purpose
  // of calculating topological charge all triangles should have the same
  // handedness.
  std::unordered_set<Triplet, TripletHasher, HandedTripletComparator> unique_indices;

  for (int spin_i = 0; spin_i < globals::num_spins; ++spin_i) {
	auto r_i = globals::lattice->atom_position(spin_i); // cartesian possition of spin i

	// Find neighbours within the same layer (assuming planes normal
	// to z). Then generate ijk sets but always moving in a clockwise direction
	// by converting the coordinates to polar angles in the plane.

	std::vector<std::pair<int, double>> nbr_polar_angles;
	for (const auto &nbr: neartree.neighbours(r_i, radius_cutoff_)) {
	  const auto r_j = nbr.first;
	  const auto spin_j = nbr.second;

	  // Only look for neighbours in the same layer
	  if (!approximately_equal(r_i[2], r_j[2], 1e-6)) {
		continue;
	  }

	  double phi = atan2(r_i[1] - r_j[1], r_i[0] - r_j[0]) + M_PI;
	  nbr_polar_angles.push_back({spin_j, phi});
	}

	// Sort by the polar angle
	sort(begin(nbr_polar_angles), end(nbr_polar_angles),
		 [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
		   return a.second < b.second;
		 });

	// Because we have sorted by polar angle the right-handed triangles which
	// start at site i simply use the increasing pairs of neighbours for j and k.
	for (auto i = 0; i < nbr_polar_angles.size() - 1; ++i) {
	  const auto spin_j = nbr_polar_angles[i].first;
	  const auto spin_k = nbr_polar_angles[i + 1].first;

	  auto result = unique_indices.insert({spin_i, spin_j, spin_k});
	}

	// Insert the final triangle which is made of the last and first neighbours.
	auto result = unique_indices
		.insert({spin_i, nbr_polar_angles.back().first,
				 nbr_polar_angles.front().first});
  }
  // Copy the set data into the vector
  triangle_indices_.clear();
  triangle_indices_.insert(triangle_indices_.end(), unique_indices.begin(), unique_indices.end());

  adjacent_triangles_.resize(globals::num_spins);
  for (auto n = 0; n < triangle_indices_.size(); ++n) {
	auto t = triangle_indices_[n];

	adjacent_triangles_[t.i].push_back(n);
	adjacent_triangles_[t.j].push_back(n);
	adjacent_triangles_[t.k].push_back(n);
  }

}
double TopologicalGeometricalDefMonitor::local_topological_charge(const Vec3 &s_i,const Vec3 &s_j,const Vec3 &s_k) const {
  double triple_product = scalar_triple_product(s_i, s_j, s_k);
  double denominator = 1 + dot(s_i, s_j) + dot(s_i, s_k) + dot(s_j, s_k);

  if (denominator <= 0.0) {
	return 0.0;
  }

  return 2.0 * atan2(triple_product, denominator) * recip_num_layers_;

}
double TopologicalGeometricalDefMonitor::local_topological_charge(const TopologicalGeometricalDefMonitor::Triplet &t) const {
  Vec3 s_i = jams::montecarlo::get_spin(t.i);
  Vec3 s_j = jams::montecarlo::get_spin(t.j);
  Vec3 s_k = jams::montecarlo::get_spin(t.k);

  return local_topological_charge(s_i, s_j, s_k);
}
double TopologicalGeometricalDefMonitor::total_topological_charge() const {
  double sum = 0.0;
  for (const auto& ijk : triangle_indices_) {
	sum += local_topological_charge(ijk);
  }

  return sum / (4.0 * kPi);
}

std::string TopologicalGeometricalDefMonitor::tsv_header(){
  using namespace jams;

  std::stringstream ss;
  ss.width(12);

  ss <<fmt::sci << "time";
  ss <<fmt::decimal << "topological_charge";

  ss << std::endl;

  return ss.str();
}


