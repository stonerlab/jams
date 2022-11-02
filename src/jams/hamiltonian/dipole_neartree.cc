#include <jams/maths/parallelepiped.h>
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "jams/hamiltonian/dipole_neartree.h"
#include "jams/interface/openmp.h"

DipoleNearTreeHamiltonian::DipoleNearTreeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size),
  r_cutoff_(jams::config_required<double>(settings, "r_cutoff")),
  neartree_(lattice->get_supercell().a(), lattice->get_supercell().b(), lattice->get_supercell().c(), lattice->periodic_boundaries(), r_cutoff_, jams::defaults::lattice_tolerance)
{

  std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  if (r_cutoff_ > lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(lattice->max_interaction_radius())  + ")");
  }

  neartree_.insert_sites(lattice->atom_cartesian_positions());


  std::cout << "  near tree size " << neartree_.size() << "\n";
  std::cout << "  near tree memory " << memory_in_natural_units(neartree_.memory()) << "\n";
}


double DipoleNearTreeHamiltonian::calculate_total_energy() {
    double e_total = 0.0;

    #pragma omp parallel for shared(globals::num_spins) default(none) reduction(+: e_total)
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
  #pragma omp parallel for shared(globals::num_spins) default(none)
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i);
  }
}


[[gnu::hot]]
Vec3 DipoleNearTreeHamiltonian::calculate_field(const int i)
{
  using namespace globals;

  const Vec3 r_i = lattice->atom_position(i);

  const auto neighbours = neartree_.neighbours(r_i, r_cutoff_);

  const double w0 = mus(i) * kVacuumPermeabilityIU / (4.0 * kPi * pow3(lattice->parameter()));
  // 2020-04-21 Using OMP on this loop gives almost no speedup because the heavy
  // work is already done to find the neighbours.

  Vec3 field = {0.0, 0.0, 0.0};
  for (const auto & neighbour : neighbours) {
    const int j = neighbour.second;
    if (j == i) continue;

    const Vec3 s_j = {s(j,0), s(j,1), s(j,2)};
    const Vec3 r_ij =  neighbour.first - r_i;

    field += w0 * mus(j) * (3.0 * r_ij * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j) / pow5(norm(r_ij));
  }
  return field;
}


void DipoleNearTreeHamiltonian::calculate_fields() {
  OMP_PARALLEL_FOR
  for (auto i = 0; i < globals::num_spins; ++i) {
        const auto field = calculate_field(i);

        for (auto n : {0,1,2}) {
            field_(i, n) = field[n];
        }
    }
}