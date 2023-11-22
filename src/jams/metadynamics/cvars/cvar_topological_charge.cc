// cvar_topological_charge.cc                                                          -*-C++-*-
#include <jams/metadynamics/cvars/cvar_topological_charge.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/lattice/interaction_neartree.h>

#include <unordered_set>
#include <vector>
#include <set>
#include <jams/helpers/montecarlo.h>
#include <jams/helpers/error.h>


jams::CVarTopologicalCharge::CVarTopologicalCharge(
    const libconfig::Setting &settings) {

  jams_warning("Metadynamics CV 'topological_charge' is based on the geometrical\n"
               "definition of topological charge and CANNOT be used as a CV.\n"
               "We have retained the module purely for demonstration of it\n"
               "not working.\n");

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
}

double jams::CVarTopologicalCharge::value() {
  return cached_value();
}


void jams::CVarTopologicalCharge::calculate_elementary_triangles(){

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


//
// Returns the topological charge of a single triangle made from the spins
// s_i, s_j and s_k using the equation
//
//    q_ijk = (s_i . (s_j x s_k))
//          / (1  + s_i.s_j + s_i.s_k + s_j.s_k)
//
// If the denominator (1  + s_i.s_j + s_i.s_k + s_j.s_k) <= 0 then the equation
// is undefined for the topological charge (basically the texture must then
// not be smooth and the denominator changes the sign of q_ijk)
double jams::CVarTopologicalCharge::local_topological_charge(const Vec3& s_i, const Vec3& s_j, const Vec3& s_k) const {
  double triple_product = scalar_triple_product(s_i, s_j, s_k);
  double denominator = 1 + dot(s_i, s_j) + dot(s_i, s_k) + dot(s_j, s_k);

  if (denominator <= 0.0) {
    return 0.0;
  }

  return 2.0 * atan2(triple_product, denominator) * recip_num_layers_;
}


//
// Returns the topological charge of a single triangle described by a triplet
// using the equation
//
//    q_ijk = (s_i . (s_j x s_k))
//          / (1  + s_i.s_j + s_i.s_k + s_j.s_k)
//
double jams::CVarTopologicalCharge::local_topological_charge(const Triplet &t) const {
  Vec3 s_i = montecarlo::get_spin(t.i);
  Vec3 s_j = montecarlo::get_spin(t.j);
  Vec3 s_k = montecarlo::get_spin(t.k);

  return local_topological_charge(s_i, s_j, s_k);
}


//
// Returns the topological charge of the system which is the sum of the local
// charge of all triangles
//
// Q = (1/4\pi) \sum_ijk q_ijk
//
double jams::CVarTopologicalCharge::total_topological_charge() const {
  double sum = 0.0;
  for (const auto& ijk : triangle_indices_) {
    sum += local_topological_charge(ijk);
  }

  return sum / (4.0 * kPi);
}


double jams::CVarTopologicalCharge::calculate_expensive_cache_value() {
  return total_topological_charge();
}


std::string jams::CVarTopologicalCharge::name() {
  return name_;
}


double jams::CVarTopologicalCharge::spin_move_trial_value(int i,
                                                          const Vec3 &spin_initial,
                                                          const Vec3 &spin_trial) {
  const double trial_value = cached_value() + topological_charge_difference(i, spin_initial, spin_trial);

  set_cache_values(i, spin_initial, spin_trial, cached_value(), trial_value);

  return trial_value;
}


double jams::CVarTopologicalCharge::topological_charge_difference(int index,
                                                                  const Vec3 &spin_initial,
                                                                  const Vec3 &spin_final) const {
  double q_ijk_initial = 0.0;
  double q_ijk_final = 0.0;

  for (const auto& n : adjacent_triangles_[index]) {
    auto t = triangle_indices_[n];

    // The 'index' spin could be any one of i,j,k so we need to check them all
    Vec3 s_i = t.i == index ? spin_initial : jams::montecarlo::get_spin(t.i);
    Vec3 s_j = t.j == index ? spin_initial : jams::montecarlo::get_spin(t.j);
    Vec3 s_k = t.k == index ? spin_initial : jams::montecarlo::get_spin(t.k);

    q_ijk_initial += local_topological_charge(s_i, s_j, s_k);

    s_i = t.i == index ? spin_final : jams::montecarlo::get_spin(t.i);
    s_j = t.j == index ? spin_final : jams::montecarlo::get_spin(t.j);
    s_k = t.k == index ? spin_final : jams::montecarlo::get_spin(t.k);

    q_ijk_final += local_topological_charge(s_i, s_j, s_k);
  }

  return (q_ijk_final - q_ijk_initial) / (4.0 * kPi);
}
