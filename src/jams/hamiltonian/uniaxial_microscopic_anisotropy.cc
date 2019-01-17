#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/hamiltonian/uniaxial_microscopic_anisotropy.h"

UniaxialMicroscopicHamiltonian::UniaxialMicroscopicHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : Hamiltonian(settings, num_spins),
          mca_order_(),
          mca_value_() {

  // don't allow mixed specification of anisotropy
  if ((settings.exists("K1") || settings.exists("K2") || settings.exists("K3"))) {
    die("UniaxialMicroscopicHamiltonian: anisotropy should only be specified in terms of d2z, d4z, d6z maybe you want UniaxialHamiltonian?");
  }

  // deal with magnetocrystalline anisotropy coefficients
  if (settings.exists("d2z")) {
    if (settings["d2z"].getLength() != lattice->num_materials()) {
      die("UniaxialHamiltonian: d2z must be specified for every material");
    }
    mca_order_.push_back(2);

    jblib::Array<double, 1> mca(num_spins, 0.0);
    for (int i = 0; i < num_spins; ++i) {
      mca(i) = double(settings["d2z"][lattice->atom_material_id(i)]) * input_unit_conversion_;
    }
    mca_value_.push_back(mca);
  }

  if (settings.exists("d4z")) {
    if (settings["d4z"].getLength() != lattice->num_materials()) {
      die("UniaxialHamiltonian: d4z must be specified for every material");
    }
    mca_order_.push_back(4);
    jblib::Array<double, 1> mca(num_spins, 0.0);
    for (int i = 0; i < num_spins; ++i) {
      mca(i) = double(settings["d4z"][lattice->atom_material_id(i)]) * input_unit_conversion_;
    }
    mca_value_.push_back(mca);
  }

  if (settings.exists("d6z")) {
    if (settings["d6z"].getLength() != lattice->num_materials()) {
      die("UniaxialHamiltonian: d6z must be specified for every material");
    }
    mca_order_.push_back(6);
    jblib::Array<double, 1> mca(num_spins, 0.0);
    for (int i = 0; i < num_spins; ++i) {
      mca(i) = double(settings["d6z"][lattice->atom_material_id(i)]) * input_unit_conversion_;
    }
    mca_value_.push_back(mca);
  }
}

double UniaxialMicroscopicHamiltonian::calculate_total_energy() {
  double e_total = 0.0;
  for (int i = 0; i < energy_.size(); ++i) {
    e_total += calculate_one_spin_energy(i);
  }
  return e_total;
}

double UniaxialMicroscopicHamiltonian::calculate_one_spin_energy(const int i) {
  double energy = 0.0;

  for (int n = 0; n < mca_order_.size(); ++n) {
    energy += mca_value_[n](i) * legendre_poly(globals::s(i, 2), mca_order_[n]);
  }

  return energy;
}

double UniaxialMicroscopicHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial,
                                                                 const Vec3 &spin_final) {
  double e_initial = 0.0;
  double e_final = 0.0;

  for (int n = 0; n < mca_order_.size(); ++n) {
    e_initial += mca_value_[n](i) * legendre_poly(spin_initial[2], mca_order_[n]);
  }

  for (int n = 0; n < mca_order_.size(); ++n) {
    e_final += mca_value_[n](i) * legendre_poly(spin_final[2], mca_order_[n]);
  }

  return e_final - e_initial;
}

void UniaxialMicroscopicHamiltonian::calculate_energies() {
  for (int i = 0; i < energy_.size(); ++i) {
    energy_[i] = calculate_one_spin_energy(i);
  }
}

void UniaxialMicroscopicHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
  const double sz = globals::s(i, 2);
  local_field[0] = 0.0;
  local_field[1] = 0.0;
  local_field[2] = 0.0;

  for (int n = 0; n < mca_order_.size(); ++n) {
    local_field[2] += -mca_value_[n](i) * legendre_dpoly(sz, mca_order_[n]);
  }
}

void UniaxialMicroscopicHamiltonian::calculate_fields() {
  field_.zero();
  for (int n = 0; n < mca_order_.size(); ++n) {
    for (int i = 0; i < field_.size(0); ++i) {
      field_(i, 2) += -mca_value_[n](i) * legendre_dpoly(globals::s(i, 2), mca_order_[n]);
    }
  }
}
