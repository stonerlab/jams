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

  r_cutoff_ = jams::config_required<jams::Real>(settings, "r_cutoff");
  dipole_prefactor_ = static_cast<jams::Real>(kVacuumPermeabilityIU / (4.0 * kPi * pow3(globals::lattice->parameter())));

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

  jams::InteractionNearTree<jams::Real> neartree(
    jams::array_cast<jams::Real>(globals::lattice->get_supercell().a1()),
    jams::array_cast<jams::Real>(globals::lattice->get_supercell().a2()),
    jams::array_cast<jams::Real>(globals::lattice->get_supercell().a3()),
    globals::lattice->periodic_boundaries(), r_cutoff_, jams::defaults::lattice_tolerance);

  std::vector<Vec3R> positions;
  positions.reserve(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i)
  {
    positions.push_back(Vec3R{globals::positions(i,0), globals::positions(i,1), globals::positions(i,2)});
  }
  neartree.insert_sites(positions);

  std::size_t max_memory_per_tensor = (sizeof(std::vector<std::pair<Vec3,int>>*) + sizeof(Vec3) + sizeof(int));

  std::cout << "  dipole neighbour list memory (estimate) "
            << memory_in_natural_units(max_memory_per_tensor * globals::num_spins * neartree.num_neighbours(
                Vec3R{globals::positions(0, 0), globals::positions(0, 1), globals::positions(0, 2)}, r_cutoff_)) << std::endl;


  int num_neighbours = 0;
  neighbour_list_.resize(globals::num_spins);
  for (auto i = 0; i < neighbour_list_.size(); ++i) {
    // All neighbours of i within the r_cutoff distance
    auto all_neighbours = neartree.neighbours(Vec3R{globals::positions(i, 0), globals::positions(i, 1), globals::positions(i, 2)}, r_cutoff_);

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

jams::Real DipoleNeighbourListHamiltonian::calculate_energy(const int i, jams::Real time) {
  Vec3R s_i = jams::array_cast<jams::Real>(Vec3{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)});
  auto field = calculate_field(i, time);
  return -0.5 * jams::dot(s_i, field);
}


jams::Real DipoleNeighbourListHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) {
  const auto field = calculate_field(i, time);
  const jams::Real e_initial = -jams::dot(spin_initial, field);
  const jams::Real e_final = -jams::dot(spin_final, field);
  return 0.5 * (e_final - e_initial);
}


[[gnu::hot]]
Vec3R DipoleNeighbourListHamiltonian::calculate_field(const int i, jams::Real time)
{
  jams::Real w0 = globals::mus(i) * dipole_prefactor_;
  Vec3R r_i = {globals::positions(i,0), globals::positions(i,1), globals::positions(i,2)};
  // 2020-04-21 Using OMP on this loop gives almost no speedup because the heavy
  // work is already done to find the neighbours.

  Vec3R field = {0.0, 0.0, 0.0};
  for (const auto & neighbour : neighbour_list_[i])
  {
    int j = neighbour.second;
    if (j == i) continue;

    Vec3R s_j = jams::array_cast<jams::Real>(Vec3{globals::s(j,0), globals::s(j,1), globals::s(j,2)});
    Vec3R r_ij =  neighbour.first - r_i;

    field += w0 * globals::mus(j) * (3.0 * r_ij * jams::dot(s_j, r_ij) -
        jams::norm_squared(r_ij) * s_j) / pow5(jams::norm(r_ij));
  }
  return field;
}

