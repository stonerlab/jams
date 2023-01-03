
#include <jams/hamiltonian/applied_field.h>
#include <jams/containers/vec3.h>
#include <jams/core/globals.h>
#include <jams/interface/config.h>

#include <iostream>

AppliedFieldHamiltonian::AppliedFieldHamiltonian(
    const libconfig::Setting &settings, unsigned int size)
    : Hamiltonian(settings, size),
      applied_b_field_(jams::config_required<Vec3>(settings, "field")) {

  std::cout << "field: " << applied_b_field_ << std::endl;

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = globals::mus(i) * applied_b_field_[j];
    }
  }
}

double AppliedFieldHamiltonian::calculate_total_energy(double time) {
  double e_total = 0.0;
  calculate_energies(time);
  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void AppliedFieldHamiltonian::calculate_energies(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}

void AppliedFieldHamiltonian::calculate_fields(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto local_field = calculate_field(i, time);
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = local_field[j];
    }
  }
}

Vec3 AppliedFieldHamiltonian::calculate_field(int i, double time) {
  return globals::mus(i) * applied_b_field_;
}

double AppliedFieldHamiltonian::calculate_energy(int i, double time) {
  auto field = calculate_field(i, time);
  return -( globals::s(i,0) * field[0]
          + globals::s(i,1) * field[1]
          + globals::s(i,2) * field[2]);
}

double AppliedFieldHamiltonian::calculate_energy_difference(int i,
                                                            const Vec3 &spin_initial,
                                                            const Vec3 &spin_final, double time) {
  const auto e_initial = -dot(spin_initial, calculate_field(i, time));
  const auto e_final = -dot(spin_final, calculate_field(i, time));

  return (e_final - e_initial);
}

const Vec3& AppliedFieldHamiltonian::b_field() const {
  return applied_b_field_;
}

void AppliedFieldHamiltonian::set_b_field(const Vec3& field) {
  applied_b_field_ = field;
}



