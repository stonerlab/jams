#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "jams/hamiltonian/dipole_neartree.h"

DipoleNearTreeHamiltonian::DipoleNearTreeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size),
is_cache_valid_(size, false){

    settings.lookupValue("r_cutoff", r_cutoff_);
    std::cout << "  r_cutoff " << r_cutoff_ << "\n";

    if (r_cutoff_ > lattice->max_interaction_radius()) {
      throw std::runtime_error(
          "r_cutoff is less than the maximum permitted interaction in the system"
          " (" + std::to_string(lattice->max_interaction_radius())  + ")");
    }

    int num_neighbours = 0;
    for (auto i = 0; i < globals::num_spins; ++i) {
      num_neighbours += lattice->num_neighbours(i, r_cutoff_);
    }

    std::cout << "  num_neighbours " << num_neighbours << "\n";
}


double DipoleNearTreeHamiltonian::calculate_total_energy() {
    double e_total = 0.0;

       for (auto i = 0; i < globals::num_spins; ++i) {
           e_total += calculate_energy(i);
       }

    return e_total;
}



double DipoleNearTreeHamiltonian::calculate_energy(const int i) {
    Vec3 s_i = {{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}};
    auto field = calculate_field(i);
    return -0.5 * dot(s_i, field);
}


double DipoleNearTreeHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    const auto field = calculate_field(i);
    const double e_initial = -dot(spin_initial, field);
    const double e_final = -dot(spin_final, field);
    return 0.5 * (e_final - e_initial);
}

void DipoleNearTreeHamiltonian::calculate_energies() {
    for (auto i = 0; i < globals::num_spins; ++i) {
        energy_(i) = calculate_energy(i);
    }
}


__attribute__((hot))
Vec3 DipoleNearTreeHamiltonian::calculate_field(const int i)
{
  using namespace globals;

  const auto neighbours = lattice->atom_neighbours(i, r_cutoff_);
  const double w0 = mus(i) * kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * pow3(lattice->parameter()));
  const Vec3 r_i = lattice->atom_position(i);
  // 2020-04-21 Using OMP on this loop gives almost no speedup because the heavy
  // work is already done to find the neighbours.

  Vec3 field = {0.0, 0.0, 0.0};
  for (const auto & neighbour : neighbours) {
    const int j = neighbour.second;
    if (j == i) continue;

    const Vec3 s_j = {s(j,0), s(j,1), s(j,2)};
    const Vec3 r_ij =  neighbour.first - r_i;

    field += w0 * mus(j) * (3.0 * r_ij * dot(s_j, r_ij) - norm_sq(r_ij) * s_j) / pow5(norm(r_ij));
  }
  return field;
}


void DipoleNearTreeHamiltonian::calculate_fields() {
    for (auto i = 0; i < globals::num_spins; ++i) {
        const auto field = calculate_field(i);

        for (auto n : {0,1,2}) {
            field_(i, n) = field[n];
        }
    }
}