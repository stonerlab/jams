#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/hamiltonian/uniaxial_microscopic_anisotropy.h"
#include <jams/helpers/exception.h>

UniaxialMicroscopicAnisotropyHamiltonian::UniaxialMicroscopicAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : Hamiltonian(settings, num_spins) {

  // don't allow mixed specification of anisotropy
  if ((settings.exists("K1") || settings.exists("K2") || settings.exists("K3"))) {
    throw jams::ConfigException(settings, "anisotropy should only be specified in terms of d2z, d4z, d6z maybe you want UniaxialHamiltonian?");
  }

  std::vector<int> cfg_mca_order;
  std::vector<std::vector<double>> cfg_mca_value;

  // deal with magnetocrystalline anisotropy coefficients
  if (settings.exists("d2z")) {
    if (settings["d2z"].getLength() != globals::lattice->num_materials()) {
      throw jams::ConfigException(settings["d2z"], "must be specified for every material");
    }
    cfg_mca_order.push_back(2);

    std::vector<double> values(settings["d2z"].getLength());
    for (auto i = 0; i < settings["d2z"].getLength(); ++i) {
      values[i] = double(settings["d2z"][i]) * input_energy_unit_conversion_;
    }

    cfg_mca_value.push_back(values);
  }

  if (settings.exists("d4z")) {
    if (settings["d4z"].getLength() != globals::lattice->num_materials()) {
      throw jams::ConfigException(settings["d4z"], "must be specified for every material");
    }
    cfg_mca_order.push_back(4);

    std::vector<double> values(settings["d4z"].getLength());
    for (auto i = 0; i < settings["d4z"].getLength(); ++i) {
      values[i] = double(settings["d4z"][i]) * input_energy_unit_conversion_;
    }

    cfg_mca_value.push_back(values);
  }

  if (settings.exists("d6z")) {
    if (settings["d6z"].getLength() != globals::lattice->num_materials()) {
      throw jams::ConfigException(settings["d6z"], "must be specified for every material");

    }
    cfg_mca_order.push_back(6);

    std::vector<double> values(settings["d6z"].getLength());
    for (auto i = 0; i < settings["d6z"].getLength(); ++i) {
      values[i] = double(settings["d6z"][i]) * input_energy_unit_conversion_;
    }

    cfg_mca_value.push_back(values);
  }

  mca_order_.resize(cfg_mca_order.size());
  for (auto i = 0; i < cfg_mca_order.size(); ++i){
    mca_order_(i) = cfg_mca_order[i];
  }

  mca_value_.resize(cfg_mca_order.size(), num_spins);
  for (auto i = 0; i < cfg_mca_order.size(); ++i){
    for (auto j = 0; j < num_spins; ++j) {
      mca_value_(i, j) = cfg_mca_value[i][globals::lattice->atom_material_id(j)];
    }
  }
}

double UniaxialMicroscopicAnisotropyHamiltonian::calculate_total_energy(double time) {
  double e_total = 0.0;
  for (int i = 0; i < energy_.size(); ++i) {
    e_total += calculate_energy(i, time);
  }
  return e_total;
}

double UniaxialMicroscopicAnisotropyHamiltonian::calculate_energy(const int i, double time) {
  double energy = 0.0;

  for (int n = 0; n < mca_order_.size(); ++n) {
    energy += mca_value_(n,i) * legendre_poly(globals::s(i, 2), mca_order_(n));
  }

  return energy;
}

double UniaxialMicroscopicAnisotropyHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                                             const Vec3 &spin_final, double time) {
  double e_initial = 0.0;
  double e_final = 0.0;

  for (int n = 0; n < mca_order_.size(); ++n) {
    e_initial += mca_value_(n,i) * legendre_poly(spin_initial[2], mca_order_(n));
  }

  for (int n = 0; n < mca_order_.size(); ++n) {
    e_final += mca_value_(n,i) * legendre_poly(spin_final[2], mca_order_(n));
  }

  return e_final - e_initial;
}

void UniaxialMicroscopicAnisotropyHamiltonian::calculate_energies(double time) {
  for (int i = 0; i < energy_.size(); ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}

Vec3 UniaxialMicroscopicAnisotropyHamiltonian::calculate_field(const int i, double time) {
  const double sz = globals::s(i, 2);
  Vec3 field = {0.0, 0.0, 0.0};

  for (int n = 0; n < mca_order_.size(); ++n) {
    field[2] += -mca_value_(n,i) * legendre_dpoly(sz, mca_order_(n));
  }
  return field;
}

void UniaxialMicroscopicAnisotropyHamiltonian::calculate_fields(double time) {
  field_.zero();
  for (int n = 0; n < mca_order_.size(); ++n) {
    for (int i = 0; i < field_.size(0); ++i) {
      field_(i, 2) += -mca_value_(n,i) * legendre_dpoly(globals::s(i, 2), mca_order_(n));
    }
  }
}
