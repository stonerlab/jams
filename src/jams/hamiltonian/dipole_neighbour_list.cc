#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "jams/hamiltonian/dipole_neighbour_list.h"
#include <jams/lattice/interaction_neartree.h>

#include <iostream>

DipoleNeighbourListHamiltonian::DipoleNeighbourListHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size){

  r_cutoff_ = jams::config_required<double>(settings, "r_cutoff");
  std::cout << "  r_cutoff " << r_cutoff_ << std::endl;

  if (r_cutoff_ > globals::lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(globals::lattice->max_interaction_radius()) + ")");
  }

  // This default predicate means every atom will be selected
  std::function<bool(const int, const int)> selection_predicate =
      [](const int i, const int j) { return true; };

  // If we choose an exclusive pair of materials then we change the predicate
  // so that only interactions a-b and b-a are calculated.
  if (settings.exists("exclusive_pair")) {
    const std::string a = settings["exclusive_pair"][0];
    if (!globals::lattice->material_exists(a)) {
      throw std::runtime_error("material " + a + " does not exist");
    }

    const std::string b = settings["exclusive_pair"][1];
    if (!globals::lattice->material_exists(b)) {
      throw std::runtime_error("material " + b + " does not exist");
    }

    // Change the Hamiltonian name to include the material pair. This is used
    // for example to name columns in output files.
    set_name(name() + "_" + a + "_" + b);

    // select materials a and b either way around
    selection_predicate = [=](const int i, const int j) {
        return (globals::lattice->lattice_site_material_name(i) == a && globals::lattice->lattice_site_material_name(j) == b)
               || (globals::lattice->lattice_site_material_name(i) == b && globals::lattice->lattice_site_material_name(j) == a);
    };
  }

  jams::InteractionNearTree neartree(globals::lattice->get_supercell().a1(),
                                     globals::lattice->get_supercell().a2(),
                                     globals::lattice->get_supercell().a3(), globals::lattice->periodic_boundaries(), r_cutoff_, jams::defaults::lattice_tolerance);
  neartree.insert_sites(globals::lattice->lattice_site_positions_cart());

  std::size_t max_memory_per_tensor = (sizeof(std::vector<std::pair<Vec3,int>>*) + sizeof(Vec3) + sizeof(int));

  std::cout << "  dipole neighbour list memory (estimate) "
            << memory_in_natural_units(max_memory_per_tensor * globals::num_spins * neartree.num_neighbours(
                globals::lattice->lattice_site_position_cart(0), r_cutoff_)) << std::endl;


  int num_neighbours = 0;
  neighbour_list_.resize(globals::num_spins);
  for (auto i = 0; i < neighbour_list_.size(); ++i) {
    // All neighbours of i within the r_cutoff distance
    auto all_neighbours = neartree.neighbours(globals::lattice->lattice_site_position_cart(i), r_cutoff_);

    // Select only the neighbours which obey the predicate
    for (const auto& nbr : all_neighbours) {
      if (selection_predicate(i, nbr.second)) {
        neighbour_list_[i].push_back(nbr);
      }
    }
    num_neighbours += neighbour_list_[i].size();
  }

  std::cout << "  num_neighbours " << num_neighbours << "\n";

  std::cout << "  dipole neighbour list memory (actual) "
       << memory_in_natural_units(max_memory_per_tensor * num_neighbours) << std::endl;
}


double DipoleNeighbourListHamiltonian::calculate_total_energy(double time) {
  double e_total = 0.0;

  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += calculate_energy(i, time);
  }

  return e_total;
}



double DipoleNeighbourListHamiltonian::calculate_energy(const int i, double time) {
  Vec3 s_i = {{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}};
  auto field = calculate_field(i, time);
  return -0.5 * dot(s_i, field);
}


double DipoleNeighbourListHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) {
  const auto field = calculate_field(i, time);
  const double e_initial = -dot(spin_initial, field);
  const double e_final = -dot(spin_final, field);
  return 0.5 * (e_final - e_initial);
}

void DipoleNeighbourListHamiltonian::calculate_energies(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}


[[gnu::hot]]
Vec3 DipoleNeighbourListHamiltonian::calculate_field(const int i, double time)
{
  double w0 = globals::mus(i) * kVacuumPermeabilityIU / (4.0 * kPi * pow3(globals::lattice->parameter()));
  Vec3 r_i = globals::lattice->lattice_site_position_cart(i);
  // 2020-04-21 Using OMP on this loop gives almost no speedup because the heavy
  // work is already done to find the neighbours.

  Vec3 field = {0.0, 0.0, 0.0};
  for (const auto & neighbour : neighbour_list_[i]) {
    int j = neighbour.second;
    if (j == i) continue;

    Vec3 s_j = {globals::s(j,0), globals::s(j,1), globals::s(j,2)};
    Vec3 r_ij =  neighbour.first - r_i;

    field += w0 * globals::mus(j) * (3.0 * r_ij * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j) / pow5(norm(r_ij));
  }
  return field;
}


void DipoleNeighbourListHamiltonian::calculate_fields(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto field = calculate_field(i, time);

    for (auto n : {0,1,2}) {
      field_(i, n) = field[n];
    }
  }
}