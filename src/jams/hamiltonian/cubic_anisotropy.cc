//
// Created by Sean Stansill [ll14s26s] on 28/10/2019.
//
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/hamiltonian/cubic_anisotropy.h"

#include <iostream>

using libconfig::Setting;
using std::vector;
using std::string;
using std::runtime_error;
// Settings should look like:
// {
//    module = "Cubic";
//    K1 = (1e-24, 2e-24); for two magnetic materials with standard cubic_anisotropy axes of [1,0,0], [0,1,0] and [0,0,1]
// }
//
// {
//    module = "cubic_anisotropy";
//    K1 = ((1e-24, [1, 0, 0], [0, 1, 0], [0, 0, 1]),
//          (2e-24, [1, 0, 0], [0, 1, 0], [0, 0, 1])); specifying the different cubic_anisotropy axes for two materials. Vectors
//          don't have to be normalized as this is done in the code
//          Can add in a check that two are orthogonal similar to the mumax implementation
// }

struct AnisotropySetting_cube {
    unsigned order;
    double coefficient;
    Vec3 axis1 = {1.0, 0.0, 0.0};
    Vec3 axis2 = {0.0, 1.0, 0.0};
    Vec3 axis3 = {0.0, 0.0, 1.0};
};

AnisotropySetting_cube read_anisotropy_setting_cube(Setting &setting, std::string order_name) {

    AnisotropySetting_cube result;

    if (order_name=="K1") {
        result.order = 1;
    } else {
        result.order = 2;
    }

    if (setting.isList()) {
        result.coefficient = setting[0];
        result.axis1 = normalize(Vec3{setting[1][0], setting[1][1], setting[1][2]});
        result.axis2 = normalize(Vec3{setting[2][0], setting[2][1], setting[2][2]});
        result.axis3 = normalize(Vec3{setting[3][0], setting[3][1], setting[3][2]});
    } else if (setting.isScalar()) {
        result.coefficient = setting;
    } else {
        throw runtime_error("Incorrectly formatted cubic_anisotropy");
    }

    return result;
}

vector<AnisotropySetting_cube> read_all_cubic_anisotropy_settings(const Setting &settings, const std::string order_name) {
    vector<AnisotropySetting_cube> anisotropies;
    for (auto i = 0; i < settings[order_name].getLength(); ++i) {
        anisotropies.push_back(read_anisotropy_setting_cube(settings[order_name][i], order_name));
    }

    return anisotropies;
}

CubicHamiltonian::CubicHamiltonian(const Setting &settings, const unsigned int num_spins)
    : Hamiltonian(settings, num_spins) {

    std::string order_name;

    if (settings.exists("K1") && settings.exists("K2")) {
        throw runtime_error("Input only one order of cubic anisotropy");
    } else if (settings.exists("K1")){
        order_name = "K1";
    } else if (settings.exists("K2")) {
        order_name = "K2";
    }
    else {
        throw runtime_error("Unsupported cubic anisotropy");
    }

    auto cubic_anisotropies = read_all_cubic_anisotropy_settings(settings, order_name);

    zero(magnitude_.resize(num_spins));
    zero(order_.resize(num_spins));
    zero(axis1_.resize(num_spins, 3));
    zero(axis2_.resize(num_spins, 3));
    zero(axis3_.resize(num_spins, 3));

    
    for (int i = 0; i < globals::num_spins; ++i) {
        for (int n = 0; n < globals::lattice->num_materials(); ++n) {
            if (globals::lattice->atom_material_id(i) == n) {
                magnitude_(i) = cubic_anisotropies[n].coefficient * input_energy_unit_conversion_;
                for (int j = 0; j < 3; ++j) {
                    axis1_(i, j) = cubic_anisotropies[n].axis1[j];
                    axis2_(i, j) = cubic_anisotropies[n].axis2[j];
                    axis3_(i, j) = cubic_anisotropies[n].axis3[j];
                }
            }
        // All anisotropies have the same order so can be done outside loop
        order_(i) = cubic_anisotropies[0].order;
        }
    }
}

double CubicHamiltonian::calculate_total_energy(double time) {
  double e_total = 0.0;
  for (auto i = 0; i < energy_.size(); ++i) {
    e_total += calculate_energy(i, time);
  }
  return e_total;
}

double CubicHamiltonian::calculate_energy(const int i, double time) {
    double energy = 0.0;

    Vec3 spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    Vec3 u = {axis1_(i, 0), axis1_(i, 1), axis1_(i, 2)};
    Vec3 v = {axis2_(i, 0), axis2_(i, 1), axis2_(i, 2)};
    Vec3 w = {axis3_(i, 0), axis3_(i, 1), axis3_(i, 2)};

    double Su2 = dot_squared(spin, u);
    double Sv2 = dot_squared(spin, v);
    double Sw2 = dot_squared(spin, w);

    if(order_(i) == 1) {
      energy += -magnitude_(i) * (Su2 * Sv2 + Sv2 * Sw2 + Sw2 * Su2);
    }
    
    if(order_(i) == 2){
      energy += -magnitude_(i) * (Su2 * Sv2 * Sw2);
    }
    
    return energy;
}

double CubicHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                     const Vec3 &spin_final, double time) {
    double e_initial = 0.0;
    double e_final = 0.0;

    Vec3 u = {axis1_(i, 0), axis1_(i, 1), axis1_(i, 2)};
    Vec3 v = {axis2_(i, 0), axis2_(i, 1), axis2_(i, 2)};
    Vec3 w = {axis3_(i, 0), axis3_(i, 1), axis3_(i, 2)};

    double Su2_initial = dot_squared(spin_initial, u);
    double Sv2_initial = dot_squared(spin_initial, v);
    double Sw2_initial = dot_squared(spin_initial, w);

    double Su2_final = dot_squared(spin_final, u);
    double Sv2_final = dot_squared(spin_final, v);
    double Sw2_final = dot_squared(spin_final, w);

    if(order_(i) == 1) {
      e_initial += -magnitude_(i) * (Su2_initial * Sv2_initial + Sv2_initial * Sw2_initial + Sw2_initial * Su2_initial);

      e_final += -magnitude_(i) * (Su2_final * Sv2_final + Sv2_final * Sw2_final + Sw2_final * Su2_final);

    }

    if(order_(i) == 2) {
      e_initial += -magnitude_(i) * (Su2_initial * Sv2_initial * Sw2_initial);

      e_final += -magnitude_(i) * (Su2_final * Sv2_final * Sw2_final);
    }

  return e_final - e_initial;
}

void CubicHamiltonian::calculate_energies(double time) {
  for (int i = 0; i < energy_.size(); ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}

Vec3 CubicHamiltonian::calculate_field(const int i, double time) {
  Vec3 field = {0.0, 0.0, 0.0};

    Vec3 spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};

    Vec3 u = {axis1_(i, 0), axis1_(i, 1), axis1_(i, 2)};
    Vec3 v = {axis2_(i, 0), axis2_(i, 1), axis2_(i, 2)};
    Vec3 w = {axis3_(i, 0), axis3_(i, 1), axis3_(i, 2)};

    double Su = dot(spin, u);
    double Sv = dot(spin, v);
    double Sw = dot(spin, w);

    auto pre = 2.0 * magnitude_(i);

    if (order_(i) == 1) {
      for (auto j = 0; j < 3; ++j) {
        field[j] += pre * ( u[j] * Su * (pow2(Sv) + pow2(Sw)) + v[j] * Sv * (pow2(Sw) + pow2(Su))
                            + w[j] * Sw * (pow2(Su) + pow2(Sv)) );
      }
    }

    if (order_(i) == 2) {
      for (auto j = 0; j < 3; ++j) {
          field[j] += pre * ( u[j] * Su * pow2(Sv) * pow2(Sw) + v[j] * Sv * pow2(Sw) * pow2(Su)
                              + w[j] * Sw * pow2(Su) * pow2(Sv) );
      }
    }

  return field;
}

void CubicHamiltonian::calculate_fields(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto field = calculate_field(i, time);
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = field[j];
    }
  }
}