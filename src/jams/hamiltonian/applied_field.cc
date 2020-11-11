
#include <jams/hamiltonian/applied_field.h>
#include <jams/containers/vec3.h>

using namespace std;

AppliedFieldHamiltonian::AppliedFieldHamiltonian(
    const libconfig::Setting &settings, unsigned int size)
    : Hamiltonian(settings, size),
      applied_field_(jams::config_required<Vec3>(settings, "field")) {

    cout << "field: " << applied_field_ << endl;

  for (int i = 0; i < globals::num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      field_(i, j) = applied_field_[j];
    }
  }
}

double AppliedFieldHamiltonian::calculate_total_energy() {
  double e_total = 0.0;
  for (int i = 0; i < globals::num_spins; ++i) {
    e_total += calculate_energy(i);
  }
  return e_total;
}

void AppliedFieldHamiltonian::calculate_energies() {
  for (int i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i);
  }
}

void AppliedFieldHamiltonian::calculate_fields() {
// do nothing - field_ is set constant in the constructor
}

Vec3 AppliedFieldHamiltonian::calculate_field(int i) {
  return applied_field_;
}

double AppliedFieldHamiltonian::calculate_energy(int i) {
  using namespace globals;
  return -( s(i,0) * applied_field_[0]
          + s(i,1) * applied_field_[1]
          + s(i,2) * applied_field_[2]);
}

double AppliedFieldHamiltonian::calculate_energy_difference(int i,
                                                            const Vec3 &spin_initial,
                                                            const Vec3 &spin_final) {
  const auto e_initial = -dot(spin_initial, applied_field_);
  const auto e_final = -dot(spin_final, applied_field_);

  return (e_final - e_initial);
}



