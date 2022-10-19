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
//    order = "K1";
//    anisotropies = (
//      (1, [0.0, 0.0, 1.0], 1e-24)
//    );
// }
//
// {
//    module = "uniaxial";
//    order = "K1";
//    anisotropies = (
//      ("Fe", [0.0, 0.0, 1.0], 1e-24)
//    );
// }

struct AnisotropySetting {
    int      motif_position = -1;
    string   material = "";
    Vec3     axis = {0.0, 0.0, 1.0};
    double   energy = 0.0;
};

int anisotropy_power_from_name(const string name) {
  if (name == "K1") return 2;
  if (name == "K2") return 4;
  if (name == "K3") return 6;
  throw runtime_error("Unsupported anisotropy: " + name);
}

AnisotropySetting read_anisotropy_setting(Setting &setting) {
  if (!setting.isList()) {
    throw runtime_error("Incorrectly formatted anisotropy setting");
  }
  AnisotropySetting result;

  if (setting[0].isNumber()) {
    result.motif_position = int(setting[0]);
    result.motif_position--;
    if (result.motif_position < 0 || result.motif_position >= lattice->num_motif_atoms()) {
      throw runtime_error("uniaxial anisotropy motif position is invalid");
    }
  } else {
    result.material = string(setting[0].c_str());
    if (!lattice->material_exists(result.material)) {
      throw runtime_error("uniaxial anisotropy material is invalid");
    }
  }
  result.axis = normalize(Vec3{setting[1][0], setting[1][1], setting[1][2]});
  result.energy = double(setting[2]);
  return result;
}

vector<AnisotropySetting> read_all_anisotropy_settings(const Setting &settings) {
  vector<AnisotropySetting> anisotropies;
  for (auto i = 0; i < settings["anisotropies"].getLength(); ++i) {
    anisotropies.push_back(read_anisotropy_setting(settings["anisotropies"][i]));
  }

  return anisotropies;
}

UniaxialHamiltonian::UniaxialHamiltonian(const Setting &settings, const unsigned int num_spins)
        : Hamiltonian(settings, num_spins) {

  // check if the old format is being used
  if (settings.exists("d2z") || settings.exists("d4z") || settings.exists("d6z")
   || settings.exists("K1") || settings.exists("K2") || settings.exists("K3")) {
    jams_die(
            "UniaxialHamiltonian: anisotropy should only be specified for a single K1, K2 or K3. "
            "For d2z, d4z, d6z you want UniaxialCoefficientHamiltonian.");
  }

  string order = jams::config_required<string>(settings, "order");
  power_ = anisotropy_power_from_name(order);

  auto anisotropies = read_all_anisotropy_settings(settings);

  zero(magnitude_.resize(num_spins));
  zero(axis_.resize(num_spins, 3));

  for (const auto& ani : anisotropies) {
    for (auto i = 0; i < globals::num_spins; ++i) {
      if (lattice->atom_motif_position(i) == ani.motif_position) {
        magnitude_(i) = ani.energy * input_energy_unit_conversion_;
        for (auto j : {0, 1, 2}) {
          axis_(i, j) = ani.axis[j];
        }
      }
      if (lattice->material_exists(ani.material)) {
        if (lattice->atom_material_id(i) == lattice->material_id(ani.material)) {
          magnitude_(i) = ani.energy * input_energy_unit_conversion_;
          for (auto j : {0, 1, 2}) {
            axis_(i, j) = ani.axis[j];
          }
        }
      }
    }
  }
}


double UniaxialHamiltonian::calculate_total_energy() {
  double e_total = 0.0;
  for (int i = 0; i < energy_.size(); ++i) {
    e_total += calculate_energy(i);
  }
  return e_total;
}

double UniaxialHamiltonian::calculate_energy(const int i) {
  using namespace globals;
  double energy = 0.0;

  auto dot = (axis_(i,0) * s(i,0) + axis_(i,1) * s(i,1) + axis_(i,2) * s(i,2));
  energy += (-magnitude_(i) * pow(dot, power_));

  return energy;
}

double UniaxialHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                        const Vec3 &spin_final) {
  double e_initial = 0.0;
  double e_final = 0.0;

  auto s_dot_a = spin_initial[0] * axis_(i,0) + spin_initial[1] * axis_(i,1) + spin_initial[2] * axis_(i,2);
  e_initial += (-magnitude_(i) * pow(s_dot_a, power_));

  auto s_dot_b = spin_final[0] * axis_(i,0) + spin_final[1] * axis_(i,1) + spin_final[2] * axis_(i,2);
  e_final += (-magnitude_(i) * pow(s_dot_b, power_));

  return e_final - e_initial;
}

void UniaxialHamiltonian::calculate_energies() {
  for (auto i = 0; i < energy_.size(); ++i) {
    energy_(i) = calculate_energy(i);
  }
}

Vec3 UniaxialHamiltonian::calculate_field(const int i) {
  using namespace globals;

  auto dot = (axis_(i,0) * s(i,0) + axis_(i,1) * s(i,1) + axis_(i,2) * s(i,2));

  Vec3 field;
  for (auto j = 0; j < 3; ++j) {
    field[j] = magnitude_(i) * power_ * pow(dot, power_ - 1) * axis_(i,j);
  }
  return field;
}

void UniaxialHamiltonian::calculate_fields() {
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto field = calculate_field(i);
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = field[j];
    }
  }
}

double UniaxialHamiltonian::calculate_internal_energy_difference(const int i) {
    using namespace globals;
    double dU = 0.0;

    const auto dot = (axis_(i,0) * s(i,0) + axis_(i,1) * s(i,1) + axis_(i,2) * s(i,2));

    const Vec3 cross = {s(i,1)*axis_(i,2) - s(i,2)*axis_(i,1), s(i,2)*axis_(i,0) - s(i,0)*axis_(i,2), s(i,0)*axis_(i,1) - s(i,1)*axis_(i,0)};
    const auto mod_cross_sq = norm_sq(cross);

    dU += power_*magnitude_(i)*( pow(dot, power_) - (power_ - 1) * mod_cross_sq*pow(dot, power_-2) );

    return dU;
}

double UniaxialHamiltonian::calculate_total_internal_energy_difference() {
    using namespace globals;
    double dU_total = 0.0;

    #if HAS_OMP
    #pragma omp parallel for default(none) shared(num_spins) reduction(+:dU_total)
    #endif
    for (int i = 0; i < num_spins; ++i) {
        dU_total += calculate_internal_energy_difference(i);
    }
    return dU_total;
}

double UniaxialHamiltonian::calculate_entropy(int i) {
    using namespace globals;
    double TS = 0.0;

    const Vec3 spin = {s(i,0), s(i,1), s(i,2)};
    const Vec3 axis = {axis_(i,0), axis_(i,1), axis_(i,2)};

    const auto dot_product = dot(spin, axis);

    const auto cross_product_norm = norm(cross(spin, axis));

    TS += power_*magnitude_(i) * pow(dot_product, power_-1) * cross_product_norm;

    return TS;
}

double UniaxialHamiltonian::calculate_total_entropy() {
    using namespace globals;
    double TS_total = 0.0;

    #if HAS_OMP
    #pragma omp parallel for default(none) shared(num_spins) reduction(+:TS_total)
    #endif
    for (int i = 0; i < num_spins; ++i) {
        TS_total += calculate_entropy(i);
    }

     return TS_total;
}