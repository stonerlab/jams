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

  supercell_matrix_ = matrix_cast<jams::Real>(globals::lattice->get_supercell().matrix());

  frac_positions_.resize(globals::num_spins);

  for (auto i = 0; i < globals::num_spins; ++i) {
    frac_positions_[i] = globals::lattice->get_supercell().inverse_matrix() * globals::lattice->lattice_site_position_cart(
        i);
  }

}

jams::Real DipoleBruteforceHamiltonian::calculate_energy(const int i, jams::Real time) {
  Vec3R s_i = array_cast<jams::Real>(Vec3{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)});
  auto field = calculate_field(i, time);
  return -0.5 * dot(s_i, field);
}

jams::Real DipoleBruteforceHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                                const Vec3 &spin_final, jams::Real time) {
  const auto field = calculate_field(i, time);
  const jams::Real e_initial = -dot(spin_initial, field);
  const jams::Real e_final = -dot(spin_final, field);
  return 0.5 * (e_final - e_initial);
}

[[gnu::hot]]
Vec3R DipoleBruteforceHamiltonian::calculate_field(const int i, jams::Real time) {
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
          globals::lattice->lattice_site_position_cart(i),
          globals::lattice->lattice_site_position_cart(j));
  };

  const jams::Real r_cut_squared = pow2(r_cutoff_);
  const jams::Real w0 = globals::mus(i) * static_cast<jams::Real>(kVacuumPermeabilityIU / (4.0 * kPi * pow3(globals::lattice->parameter())));

  jams::Real hx = 0, hy = 0, hz = 0;
  #if HAS_OMP
  #pragma omp parallel for reduction(+:hx, hy, hz)
  #endif
  for (auto j = 0; j < globals::num_spins; ++j) {
    if (j == i) continue;

    const auto s_j = array_cast<jams::Real>(Vec3{globals::s(j,0), globals::s(j,1), globals::s(j,2)});
    auto r_ij = array_cast<jams::Real>(displacement(i, j));

    const jams::Real r_abs_sq = norm_squared(r_ij);

    const jams::Real eps = jams::defaults::lattice_tolerance;
    if (definately_greater_than(r_abs_sq, r_cut_squared, eps)) continue;
    hx += w0 * globals::mus(j) * (3.0 * r_ij[0] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[0]) / pow5(norm(r_ij));
    hy += w0 * globals::mus(j) * (3.0 * r_ij[1] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[1]) / pow5(norm(r_ij));;
    hz += w0 * globals::mus(j) * (3.0 * r_ij[2] * dot(s_j, r_ij) -
        norm_squared(r_ij) * s_j[2]) / pow5(norm(r_ij));;
  }

  return {hx, hy, hz};
}
