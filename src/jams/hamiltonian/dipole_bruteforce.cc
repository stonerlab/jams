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

#include "jams/hamiltonian/dipole_bruteforce.h"

DipoleBruteforceHamiltonian::DipoleBruteforceHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size) {

  settings.lookupValue("r_cutoff", r_cutoff_);
  std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  if (r_cutoff_ > lattice->max_interaction_radius()) {
    throw std::runtime_error(
        "r_cutoff is less than the maximum permitted interaction in the system"
        " (" + std::to_string(lattice->max_interaction_radius())  + ")");
  }

  supercell_matrix_ = lattice->get_supercell().matrix();

  frac_positions_.resize(globals::num_spins);

  for (auto i = 0; i < globals::num_spins; ++i) {
    frac_positions_[i] = lattice->get_supercell().inverse_matrix() * lattice->atom_position(i);
  }

}

double DipoleBruteforceHamiltonian::calculate_total_energy() {
  double e_total = 0.0;

  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += calculate_one_spin_energy(i);
  }

  return e_total;
}

double DipoleBruteforceHamiltonian::calculate_one_spin_energy(const int i) {
  Vec3 s_i = {{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}};
  auto field = calculate_one_spin_field(i);
  return -0.5 * dot(s_i, field);
}

double DipoleBruteforceHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial,
                                                                         const Vec3 &spin_final) {
  const auto field = calculate_one_spin_field(i);
  const double e_initial = -dot(spin_initial, field);
  const double e_final = -dot(spin_final, field);
  return 0.5 * (e_final - e_initial);
}

void DipoleBruteforceHamiltonian::calculate_energies() {
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_one_spin_energy(i);
  }
}


__attribute__((hot))
Vec3 DipoleBruteforceHamiltonian::calculate_one_spin_field(const int i) {
  using namespace globals;

  const auto r_cut_squared = pow2(r_cutoff_);
  const double w0 = mus(i) * kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * pow3(lattice->parameter()));

  double hx = 0, hy = 0, hz = 0;
  #if HAS_OMP
  #pragma omp parallel for reduction(+:hx, hy, hz)
  #endif
  for (auto j = 0; j < globals::num_spins; ++j) {
    if (j == i) continue;

    const Vec3 s_j = {s(j,0), s(j,1), s(j,2)};
    Vec3 r_ij = lattice->displacement(i, j);

    const auto r_abs_sq = norm_sq(r_ij);

    if (r_abs_sq > r_cut_squared) continue;

    const auto sj_dot_r = s(j, 0) * r_ij[0] + s(j, 1) * r_ij[1] + s(j, 2) * r_ij[2];

    hx += w0 * mus(j) * (3.0 * r_ij[0] * dot(s_j, r_ij) - pow2(norm(r_ij)) * s_j[0]) / pow5(norm(r_ij));
    hy += w0 * mus(j) * (3.0 * r_ij[1] * dot(s_j, r_ij) - pow2(norm(r_ij)) * s_j[1]) / pow5(norm(r_ij));;
    hz += w0 * mus(j) * (3.0 * r_ij[2] * dot(s_j, r_ij) - pow2(norm(r_ij)) * s_j[2]) / pow5(norm(r_ij));;
  }

  return {hx, hy, hz};
}

void DipoleBruteforceHamiltonian::calculate_fields() {
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto field = calculate_one_spin_field(i);

    for (auto n : {0,1,2}) {
      field_(i, n) = field[n];
    }
  }
}