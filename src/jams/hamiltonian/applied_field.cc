
#include <jams/hamiltonian/applied_field.h>
#include <jams/containers/vec3.h>

using namespace std;

AppliedFieldHamiltonian::AppliedFieldHamiltonian(
    const libconfig::Setting &settings, unsigned int size)
    : Hamiltonian(settings, size),
      applied_b_field_(jams::config_required<Vec3>(settings, "field")) {

    cout << "field: " << applied_b_field_ << endl;

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = globals::mus(i) * applied_b_field_[j];
    }
  }
}

double AppliedFieldHamiltonian::calculate_total_energy() {
  double e_total = 0.0;
  calculate_energies();
  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void AppliedFieldHamiltonian::calculate_energies() {
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i);
  }
}

void AppliedFieldHamiltonian::calculate_fields() {
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto local_field = calculate_field(i);
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = local_field[j];
    }
  }
}

Vec3 AppliedFieldHamiltonian::calculate_field(int i) {
  return globals::mus(i) * applied_b_field_;
}

double AppliedFieldHamiltonian::calculate_energy(int i) {
  using namespace globals;
  auto field = calculate_field(i);
  return -(s(i,0) * field[0]
          + s(i,1) * field[1]
          + s(i,2) * field[2]);
}

double AppliedFieldHamiltonian::calculate_energy_difference(int i,
                                                            const Vec3 &spin_initial,
                                                            const Vec3 &spin_final) {
  const auto e_initial = -dot(spin_initial, calculate_field(i));
  const auto e_final = -dot(spin_final, calculate_field(i));

  return (e_final - e_initial);
}

const Vec3& AppliedFieldHamiltonian::b_field() const {
  return applied_b_field_;
}

void AppliedFieldHamiltonian::set_b_field(const Vec3& field) {
  applied_b_field_ = field;
}



