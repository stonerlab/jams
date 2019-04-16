#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/hamiltonian/uniaxial_anisotropy.h"

using libconfig::Setting;
using std::vector;
using std::string;
using std::runtime_error;
// Settings should look like:
// {
//    module = "uniaxial";
//    K1 = (1e-24, 2e-24);
// }
//
// {
//    module = "uniaxial";
//    K1 = ((1e-24, [0, 0, 1]),
//          (2e-24, [0, 0, 1]));
// }

struct AnisotropySetting {
    unsigned power;
    double   energy;
    Vec3     axis;
};

unsigned anisotropy_power_from_name(const string name) {
  if (name == "K1") return 2;
  if (name == "K2") return 4;
  if (name == "K3") return 6;
  throw runtime_error("Unsupported anisotropy: " + name);
}

AnisotropySetting read_anisotropy_setting(Setting &setting) {
  if (setting.isList()) {
    Vec3 axis = {setting[1][0], setting[1][1], setting[1][2]};
    return {anisotropy_power_from_name(setting.getParent().getName()), setting[0], normalize(axis)};
  }
  if (setting.isScalar()) {
    return {anisotropy_power_from_name(setting.getParent().getName()), setting, {0.0, 0.0, 1.0}};
  }
  throw runtime_error("Incorrectly formatted anisotropy setting");
}

vector<vector<AnisotropySetting>> read_all_anisotropy_settings(const Setting &settings) {
vector<vector<AnisotropySetting>> anisotropies(lattice->num_materials());
  auto anisotropy_names = {"K1", "K2", "K3"};
  for (const auto name : anisotropy_names) {
    if (settings.exists(name)) {
      if (settings[name].getLength() != lattice->num_materials()) {
        throw runtime_error("UniaxialHamiltonian: " + string(name) + "  must be specified for every material");
      }

      for (auto i = 0; i < settings[name].getLength(); ++i) {
        anisotropies[i].push_back(read_anisotropy_setting(settings[name][i]));
      }
    }
  }
  // the array indicies are (type, power)
  return anisotropies;
}

UniaxialHamiltonian::UniaxialHamiltonian(const Setting &settings, const unsigned int num_spins)
        : Hamiltonian(settings, num_spins) {

  // check if the old format is being used
  if ((settings.exists("d2z") || settings.exists("d4z") || settings.exists("d6z"))) {
    jams_die(
            "UniaxialHamiltonian: anisotropy should only be specified in terms of K1, K2, K3 maybe you want UniaxialCoefficientHamiltonian?");
  }

  auto anisotropies = read_all_anisotropy_settings(settings);

  for (auto type = 0; type < lattice->num_materials(); ++type) {
    std::cout << "  " << lattice->material_name(type) << ":\n";
    for (const auto& ani : anisotropies[type]) {
      std::cout << "    " << ani.axis << "  " << ani.power << "  " << ani.energy << "\n";
    }
  }

  num_coefficients_ = anisotropies[0].size();

  power_.resize(num_spins, anisotropies[0].size());
  axis_.resize(num_spins, anisotropies[0].size());
  magnitude_.resize(num_spins, anisotropies[0].size());

  for (int i = 0; i < globals::num_spins; ++i) {
    auto type = lattice->atom_material_id(i);
    for (auto j = 0; j < anisotropies[type].size(); ++j) {
      power_(i, j) = anisotropies[type][j].power;
      axis_(i, j) = anisotropies[type][j].axis;
      magnitude_(i, j) = anisotropies[type][j].energy * input_unit_conversion_;
    }
  }
}


double UniaxialHamiltonian::calculate_total_energy() {
  double e_total = 0.0;
  for (int i = 0; i < energy_.size(); ++i) {
    e_total += calculate_one_spin_energy(i);
  }
  return e_total;
}

double UniaxialHamiltonian::calculate_one_spin_energy(const int i) {
  using namespace globals;
  double energy = 0.0;

  for (auto n = 0; n < num_coefficients_; ++n) {
    auto dot = (axis_(i,n)[0] * s(i,0) + axis_(i,n)[1] * s(i,1) + axis_(i,n)[2] * s(i,2));
    energy += (-magnitude_(i,n) * pow(dot, power_(i,n)));
  }

  return energy;
}

double UniaxialHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial,
                                                                 const Vec3 &spin_final) {
  double e_initial = 0.0;
  double e_final = 0.0;

  for (auto n = 0; n < num_coefficients_; ++n) {
    e_initial += (-magnitude_(i,n) * pow(dot(spin_initial, axis_(i,n)), power_(i,n)));
  }

  for (auto n = 0; n < num_coefficients_; ++n) {
    e_final += (-magnitude_(i,n) * pow(dot(spin_final, axis_(i,n)), power_(i,n)));
  }

  return e_final - e_initial;
}

void UniaxialHamiltonian::calculate_energies() {
  for (int i = 0; i < energy_.size(); ++i) {
    energy_[i] = calculate_one_spin_energy(i);
  }
}

void UniaxialHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
  using namespace globals;
  local_field[0] = 0.0;
  local_field[1] = 0.0;
  local_field[2] = 0.0;

  for (auto n = 0; n < num_coefficients_; ++n) {
    auto dot = (axis_(i,n)[0] * s(i,0) + axis_(i,n)[1] * s(i,1) + axis_(i,n)[2] * s(i,2));
    for (auto j = 0; j < 3; ++j) {
      local_field[j] += magnitude_(i,n) * power_(i,n) * pow(dot, power_(i,n) - 1) * axis_(i, n)[j];
    }
  }
}

void UniaxialHamiltonian::calculate_fields() {
  using namespace globals;
  field_.zero();

  for (auto i = 0; i < num_spins; ++i) {
    for (auto n = 0; n < num_coefficients_; ++n) {
      auto dot = (axis_(i, n)[0] * s(i, 0) + axis_(i, n)[1] * s(i, 1) + axis_(i, n)[2] * s(i, 2));
      for (auto j = 0; j < 3; ++j) {
        field_(i,j) += magnitude_(i, n) * power_(i, n) * pow(dot, power_(i, n) - 1) * axis_(i, n)[j];
      }
    }
  }

}
