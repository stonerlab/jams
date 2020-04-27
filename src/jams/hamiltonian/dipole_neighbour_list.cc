#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "jams/hamiltonian/dipole_neighbour_list.h"

DipoleNeighbourListHamiltonian::DipoleNeighbourListHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size){

  settings.lookupValue("r_cutoff", r_cutoff_);
  std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  if (r_cutoff_ > lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(lattice->max_interaction_radius())  + ")");
  }

  std::size_t max_memory_per_tensor = (sizeof(std::vector<std::pair<Vec3,int>>*) + sizeof(Vec3) + sizeof(int));

  std::cout << "  dipole neighbour list memory (estimate) "
            << memory_in_natural_units(max_memory_per_tensor * globals::num_spins * lattice->num_neighbours(0, r_cutoff_)) << std::endl;

  int num_neighbours = 0;
  neighbour_list_.resize(globals::num_spins);
  for (auto i = 0; i < neighbour_list_.size(); ++i) {
    neighbour_list_[i] = lattice->atom_neighbours(i, r_cutoff_);
    num_neighbours += neighbour_list_[i].size();
  }

  std::cout << "  num_neighbours " << num_neighbours << "\n";

  std::cout << "  dipole neighbour list memory (actual) "
       << memory_in_natural_units(max_memory_per_tensor * num_neighbours) << std::endl;
}


double DipoleNeighbourListHamiltonian::calculate_total_energy() {
  double e_total = 0.0;

  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += calculate_energy(i);
  }

  return e_total;
}



double DipoleNeighbourListHamiltonian::calculate_energy(const int i) {
  Vec3 s_i = {{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}};
  auto field = calculate_field(i);
  return -0.5 * dot(s_i, field);
}


double DipoleNeighbourListHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
  const auto field = calculate_field(i);
  const double e_initial = -dot(spin_initial, field);
  const double e_final = -dot(spin_final, field);
  return 0.5 * (e_final - e_initial);
}

void DipoleNeighbourListHamiltonian::calculate_energies() {
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i);
  }
}


__attribute__((hot))
Vec3 DipoleNeighbourListHamiltonian::calculate_field(const int i)
{
  using namespace globals;

  double w0 = mus(i) * kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * pow3(lattice->parameter()));
  Vec3 r_i = lattice->atom_position(i);
  // 2020-04-21 Using OMP on this loop gives almost no speedup because the heavy
  // work is already done to find the neighbours.

  Vec3 field = {0.0, 0.0, 0.0};
  for (const auto & neighbour : neighbour_list_[i]) {
    int j = neighbour.second;
    if (j == i) continue;

    Vec3 s_j = {s(j,0), s(j,1), s(j,2)};
    Vec3 r_ij =  neighbour.first - r_i;

    field += w0 * mus(j) * (3.0 * r_ij * dot(s_j, r_ij) - norm_sq(r_ij) * s_j) / pow5(norm(r_ij));
  }
  return field;
}


void DipoleNeighbourListHamiltonian::calculate_fields() {
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto field = calculate_field(i);

    for (auto n : {0,1,2}) {
      field_(i, n) = field[n];
    }
  }
}