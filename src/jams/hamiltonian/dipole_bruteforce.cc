//
// Created by Joseph Barker on 2020-04-26.
//

#ifdef HAS_OMP
#include <omp.h>
#endif

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include <jams/lattice/minimum_image.h>

#include "jams/hamiltonian/dipole_bruteforce.h"

#include <iostream>

DipoleBruteforceHamiltonian::DipoleBruteforceHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size) {

  settings.lookupValue("r_cutoff", r_cutoff_);
  std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  if (r_cutoff_ > globals::lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(globals::lattice->max_interaction_radius()) + ")");
  }

  supercell_matrix_ = globals::lattice->get_supercell().matrix();

  frac_positions_.resize(globals::num_spins);

  for (auto i = 0; i < globals::num_spins; ++i) {
    frac_positions_[i] = globals::lattice->get_supercell().inverse_matrix() * globals::lattice->atom_position(i);
  }

}

double DipoleBruteforceHamiltonian::calculate_total_energy(double time) {
  double e_total = 0.0;

  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += calculate_energy(i, time);
  }

  return e_total;
}

double DipoleBruteforceHamiltonian::calculate_energy(const int i, double time) {
  Vec3 s_i = {{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}};
  auto field = calculate_field(i, time);
  return -0.5 * dot(s_i, field);
}

double DipoleBruteforceHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                                const Vec3 &spin_final, double time) {
  const auto field = calculate_field(i, time);
  const double e_initial = -dot(spin_initial, field);
  const double e_final = -dot(spin_final, field);
  return 0.5 * (e_final - e_initial);
}

void DipoleBruteforceHamiltonian::calculate_energies(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}


[[gnu::hot]]
Vec3 DipoleBruteforceHamiltonian::calculate_field(const int i, double time) {
  using namespace std::placeholders;


  // We will use Smith's algorithm for the minimum image convention below which is only valid for
  // displacements less than the inradius of the cell. Our r_cutoff_ is checked at runtime in the
  // constructor for this condition which allows us to turn off the safety check in Smith's algorithm
  // (an optimisation). We assert the condition here again for safety.
  assert(r_cutoff_ <= globals::lattice->max_interaction_radius());

  auto displacement = [](const int i, const int j) {
      return jams::minimum_image_smith_method(
          globals::lattice->get_supercell().matrix(),
          globals::lattice->get_supercell().inverse_matrix(),
          globals::lattice->get_supercell().periodic(),
          globals::lattice->atom_position(i),
          globals::lattice->atom_position(j));
  };

  const auto r_cut_squared = pow2(r_cutoff_);
  const double w0 = globals::mus(i) * kVacuumPermeabilityIU / (4.0 * kPi * pow3(globals::lattice->parameter()));

  double hx = 0, hy = 0, hz = 0;
  #if HAS_OMP
  #pragma omp parallel for reduction(+:hx, hy, hz)
  #endif
  for (auto j = 0; j < globals::num_spins; ++j) {
    if (j == i) continue;

    const Vec3 s_j = {globals::s(j,0), globals::s(j,1), globals::s(j,2)};

    Vec3 r_ij = displacement(i, j);

    const auto r_abs_sq = norm_squared(r_ij);

    if (definately_greater_than(r_abs_sq, r_cut_squared, jams::defaults::lattice_tolerance)) continue;
    hx += w0 * globals::mus(j) * (3.0 * r_ij[0] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[0]) / pow5(norm(r_ij));
    hy += w0 * globals::mus(j) * (3.0 * r_ij[1] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[1]) / pow5(norm(r_ij));;
    hz += w0 * globals::mus(j) * (3.0 * r_ij[2] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[2]) / pow5(norm(r_ij));;
  }

  return {hx, hy, hz};
}

void DipoleBruteforceHamiltonian::calculate_fields(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto field = calculate_field(i, time);

    for (auto n : {0,1,2}) {
      field_(i, n) = field[n];
    }
  }
}