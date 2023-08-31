
#include "jams/hamiltonian/landau.h"
#include <jams/core/globals.h>
#include <jams/core/lattice.h>


LandauHamiltonian::LandauHamiltonian(const libconfig::Setting &settings,
                                     unsigned int size)
    : Hamiltonian(settings, size) {
  landau_A_.resize(globals::num_spins);
  landau_B_.resize(globals::num_spins);
  landau_C_.resize(globals::num_spins);

  for (int i = 0; i < globals::num_spins; ++i) {
    landau_A_(i) = double(settings["A"][globals::lattice->atom_material_id(i)]) * input_energy_unit_conversion_;
    landau_B_(i) = double(settings["B"][globals::lattice->atom_material_id(i)]) * input_energy_unit_conversion_;
    landau_C_(i) = double(settings["C"][globals::lattice->atom_material_id(i)]) * input_energy_unit_conversion_;
  }
}


double LandauHamiltonian::calculate_total_energy(double time) {
  calculate_energies(time);
  double e_total = 0.0;
  for (auto i = 0; i < energy_.size(); ++i) {
    e_total += energy_(i);
  }
  return e_total;
}


Vec3 LandauHamiltonian::calculate_field(int i, double time) {
  Vec3 spin = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
  Vec3 h = {0, 0, 0};
  double s_sq = dot(spin, spin);

  return -2.0 * landau_A_(i) * spin - 4.0 * landau_B_(i) * spin * s_sq - 6.0 * landau_C_(i) * spin * pow2(s_sq);
}


double LandauHamiltonian::calculate_energy(int i, double time) {
  double s_sq = globals::s(i,0)*globals::s(i,0) + globals::s(i,1)*globals::s(i,1) + globals::s(i,2)*globals::s(i,2);
  return landau_A_(i) * s_sq + landau_B_(i) * pow2(s_sq) + landau_C_(i) * pow3(s_sq);
}


void LandauHamiltonian::calculate_energies(double time) {
  for (int i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}


void LandauHamiltonian::calculate_fields(double time) {
  for (int i = 0; i < globals::num_spins; ++i) {
    Vec3 h = calculate_field(i, time);
    for (int n = 0; n < 3; ++n) {
      field_(i, n) = h[n];
    }
  }
}


double
LandauHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                               const Vec3 &spin_final,
                                               double time) {
  double s_initial_sq  = dot(spin_initial, spin_initial);
  double s_final_sq = dot(spin_final, spin_final);

  double e_initial = landau_A_(i) * s_initial_sq + landau_B_(i) * pow2(s_initial_sq) + landau_C_(i) * pow3(s_initial_sq);
  double e_final = landau_A_(i) * s_final_sq + landau_B_(i) * pow2(s_final_sq) + landau_C_(i) * pow3(s_final_sq);

  return e_final - e_initial;
}


