#include <jams/maths/parallelepiped.h>
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "jams/hamiltonian/dipole_neartree.h"
#include "jams/interface/openmp.h"

#include <iostream>

DipoleNearTreeHamiltonian::DipoleNearTreeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size),
  r_cutoff_(jams::config_required<double>(settings, "r_cutoff")),
  neartree_(globals::lattice->get_supercell().a1(),
            globals::lattice->get_supercell().a2(),
            globals::lattice->get_supercell().a3(), globals::lattice->periodic_boundaries(), r_cutoff_, jams::defaults::lattice_tolerance)
{

  std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  if (r_cutoff_ > globals::lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(globals::lattice->max_interaction_radius()) + ")");
  }

  neartree_.insert_sites(globals::lattice->lattice_site_positions_cart());


  std::cout << "  near tree size " << neartree_.size() << "\n";
  std::cout << "  near tree memory " << memory_in_natural_units(neartree_.memory()) << "\n";
}


double DipoleNearTreeHamiltonian::calculate_total_energy(double time) {
    double e_total = 0.0;

    #pragma omp parallel for shared(globals::num_spins) default(none) reduction(+: e_total)
    for (auto i = 0; i < globals::num_spins; ++i) {
       e_total += calculate_energy(i, time);
    }

    return e_total;
}



double DipoleNearTreeHamiltonian::calculate_energy(const int i, double time) {
    Vec3 s_i = {{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}};
    auto field = calculate_field(i, time);
    return -0.5 * dot(s_i, field);
}


double DipoleNearTreeHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) {
    const auto field = calculate_field(i, time);
    const double e_initial = -dot(spin_initial, field);
    const double e_final = -dot(spin_final, field);
    return 0.5 * (e_final - e_initial);
}

void DipoleNearTreeHamiltonian::calculate_energies(double time) {
  #pragma omp parallel for shared(globals::num_spins) default(none)
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}


[[gnu::hot]]
Vec3 DipoleNearTreeHamiltonian::calculate_field(const int i, double time)
{
  const Vec3 r_i = globals::lattice->lattice_site_position_cart(i);

  const auto neighbours = neartree_.neighbours(r_i, r_cutoff_);

  const double w0 = globals::mus(i) * kVacuumPermeabilityIU / (4.0 * kPi * pow3(globals::lattice->parameter()));
  // 2020-04-21 Using OMP on this loop gives almost no speedup because the heavy
  // work is already done to find the neighbours.

  Vec3 field = {0.0, 0.0, 0.0};
  for (const auto & neighbour : neighbours) {
    const int j = neighbour.second;
    if (j == i) continue;

    const Vec3 s_j = {globals::s(j,0), globals::s(j,1), globals::s(j,2)};
    const Vec3 r_ij =  neighbour.first - r_i;

    field += w0 * globals::mus(j) * (3.0 * r_ij * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j) / pow5(norm(r_ij));
  }
  return field;
}


void DipoleNearTreeHamiltonian::calculate_fields(double time) {
  OMP_PARALLEL_FOR
  for (auto i = 0; i < globals::num_spins; ++i) {
        const auto field = calculate_field(i, time);

        for (auto n : {0,1,2}) {
            field_(i, n) = field[n];
        }
    }
}