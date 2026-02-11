#include <jams/maths/parallelepiped.h>
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "jams/hamiltonian/dipole_neartree.h"
#include "jams/interface/openmp.h"
#include "jams/lattice/interaction_neartree.h"

#include <iostream>

DipoleNearTreeHamiltonian::DipoleNearTreeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size),
  r_cutoff_(jams::config_required<jams::Real>(settings, "r_cutoff")),
  neartree_(jams::array_cast<jams::Real>(globals::lattice->get_supercell().a1()),
            jams::array_cast<jams::Real>(globals::lattice->get_supercell().a2()),
            jams::array_cast<jams::Real>(globals::lattice->get_supercell().a3()),
            globals::lattice->periodic_boundaries(),
            r_cutoff_, jams::defaults::lattice_tolerance)
{

  std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  if (r_cutoff_ > globals::lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(globals::lattice->max_interaction_radius()) + ")");
  }

  std::vector<Vec3R> positions;
    positions.reserve(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i)
  {
      positions.push_back(jams::array_cast<jams::Real>(Vec3{globals::positions(i,0), globals::positions(i,1), globals::positions(i,2)}));
  }
  neartree_.insert_sites(positions);


  std::cout << "  near tree size " << neartree_.size() << "\n";
  std::cout << "  near tree memory " << memory_in_natural_units(neartree_.memory()) << "\n";
}


jams::Real DipoleNearTreeHamiltonian::calculate_energy(const int i, jams::Real time) {
    Vec3 s_i = {{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}};
    auto field = calculate_field(i, time);
    return -0.5 * jams::dot(s_i, field);
}


jams::Real DipoleNearTreeHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) {
    const auto field = calculate_field(i, time);
    const jams::Real e_initial = -jams::dot(spin_initial, field);
    const jams::Real e_final = -jams::dot(spin_final, field);
    return 0.5 * (e_final - e_initial);
}


[[gnu::hot]]
Vec3R DipoleNearTreeHamiltonian::calculate_field(const int i, jams::Real time)
{
  const Vec3R r_i = {globals::positions(i, 0), globals::positions(i, 1), globals::positions(i, 2)};

  const auto neighbours = neartree_.neighbours(r_i, r_cutoff_);

  const jams::Real w0 = globals::mus(i) * static_cast<jams::Real>(kVacuumPermeabilityIU / (4.0 * kPi * pow3(globals::lattice->parameter())));
  // 2020-04-21 Using OMP on this loop gives almost no speedup because the heavy
  // work is already done to find the neighbours.

  Vec3R field = {0.0, 0.0, 0.0};
  for (const auto & neighbour : neighbours) {
    const int j = neighbour.second;
    if (j == i) continue;

    const Vec3R s_j = jams::array_cast<jams::Real>(Vec3{globals::s(j,0), globals::s(j,1), globals::s(j,2)});
    const Vec3R r_ij =  neighbour.first - r_i;

    field += w0 * globals::mus(j) * (3.0 * r_ij * jams::dot(s_j, r_ij) -
        jams::norm_squared(r_ij) * s_j) / pow5(jams::norm(r_ij));
  }
  return field;
}
