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

struct CubicAnisotropySetting {
    unsigned order = 1;
    double energy = 0.0;
    Vec3 u = {1.0, 0.0, 0.0};
    Vec3 v = {0.0, 1.0, 0.0};
    Vec3 w = {0.0, 0.0, 1.0};
};

CubicAnisotropySetting read_anisotropy_setting_cube(Setting &setting, std::string order_name) {

    CubicAnisotropySetting result;

    if (order_name=="K1") {
      result.order = 1;
    } else if (order_name == "K2") {
      result.order = 2;
    } else {
      throw runtime_error("Unknown cubic anisotropy order: " + order_name);
    }

    if (setting.isList()) {
        result.energy = setting[0];
        result.u = jams::normalize(Vec3{setting[1][0], setting[1][1], setting[1][2]});
        result.v = jams::normalize(Vec3{setting[2][0], setting[2][1], setting[2][2]});
        result.w = jams::normalize(Vec3{setting[3][0], setting[3][1], setting[3][2]});
    } else if (setting.isScalar()) {
        result.energy = setting;
    } else {
        throw runtime_error("Incorrectly formatted cubic anisotropy");
    }


  // The three axes must be orthogonal and normalised. We normalised when we read the input
  // but the orthogonality must be checked.
  if (!approximately_zero(jams::dot(result.u, result.v), jams::defaults::lattice_tolerance)
  || !approximately_zero(jams::dot(result.v, result.w), jams::defaults::lattice_tolerance)
  || !approximately_zero(jams::dot(result.w, result.u), jams::defaults::lattice_tolerance))
  {
    throw runtime_error("Cubic anisotropy UVW axes must be orthogonal");
  }

  return result;
}

vector<CubicAnisotropySetting> read_all_cubic_anisotropy_settings(const Setting &settings, const std::string order_name) {
    vector<CubicAnisotropySetting> anisotropies;
    for (auto i = 0; i < settings[order_name].getLength(); ++i) {
        anisotropies.push_back(read_anisotropy_setting_cube(settings[order_name][i], order_name));
    }

    return anisotropies;
}

CubicAnisotropyHamiltonian::CubicAnisotropyHamiltonian(const Setting &settings, const unsigned int num_spins)
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
    zero(u_axes_.resize(num_spins, 3));
    zero(v_axes_.resize(num_spins, 3));
    zero(w_axes_.resize(num_spins, 3));

    
    for (int i = 0; i < globals::num_spins; ++i) {
        for (int n = 0; n < globals::lattice->num_materials(); ++n) {
            if (globals::lattice->lattice_site_material_id(i) == n) {
                magnitude_(i) = cubic_anisotropies[n].energy * input_energy_unit_conversion_;
                for (int j = 0; j < 3; ++j) {
                  u_axes_(i, j) = cubic_anisotropies[n].u[j];
                  v_axes_(i, j) = cubic_anisotropies[n].v[j];
                  w_axes_(i, j) = cubic_anisotropies[n].w[j];
                }
            }
        // All anisotropies have the same order so can be done outside loop
        order_(i) = cubic_anisotropies[0].order;
        }
    }
}


jams::Real CubicAnisotropyHamiltonian::calculate_energy(const int i, jams::Real time) {
    double energy = 0.0;

    Vec3 spin = Vec3{globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    Vec3 u = {u_axes_(i, 0), u_axes_(i, 1), u_axes_(i, 2)};
    Vec3 v = {v_axes_(i, 0), v_axes_(i, 1), v_axes_(i, 2)};
    Vec3 w = {w_axes_(i, 0), w_axes_(i, 1), w_axes_(i, 2)};

    double Su2 = jams::dot_squared(spin, u);
    double Sv2 = jams::dot_squared(spin, v);
    double Sw2 = jams::dot_squared(spin, w);

    if(order_(i) == 1) {
      energy += -magnitude_(i) * (Su2 * Sv2 + Sv2 * Sw2 + Sw2 * Su2);
    }
    
    if(order_(i) == 2){
      energy += -magnitude_(i) * (Su2 * Sv2 * Sw2);
    }
    
    return static_cast<jams::Real>(energy);
}

jams::Real CubicAnisotropyHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                               const Vec3 &spin_final, jams::Real time) {
    double e_initial = 0.0;
    double e_final = 0.0;

    Vec3 u = {u_axes_(i, 0), u_axes_(i, 1), u_axes_(i, 2)};
    Vec3 v = {v_axes_(i, 0), v_axes_(i, 1), v_axes_(i, 2)};
    Vec3 w = {w_axes_(i, 0), w_axes_(i, 1), w_axes_(i, 2)};

    double Su2_initial = jams::dot_squared(spin_initial, u);
    double Sv2_initial = jams::dot_squared(spin_initial, v);
    double Sw2_initial = jams::dot_squared(spin_initial, w);

    double Su2_final = jams::dot_squared(spin_final, u);
    double Sv2_final = jams::dot_squared(spin_final, v);
    double Sw2_final = jams::dot_squared(spin_final, w);

    if(order_(i) == 1) {
      e_initial += -magnitude_(i) * (Su2_initial * Sv2_initial + Sv2_initial * Sw2_initial + Sw2_initial * Su2_initial);

      e_final += -magnitude_(i) * (Su2_final * Sv2_final + Sv2_final * Sw2_final + Sw2_final * Su2_final);

    }

    if(order_(i) == 2) {
      e_initial += -magnitude_(i) * (Su2_initial * Sv2_initial * Sw2_initial);

      e_final += -magnitude_(i) * (Su2_final * Sv2_final * Sw2_final);
    }

  return static_cast<jams::Real>(e_final - e_initial);
}


Vec3R CubicAnisotropyHamiltonian::calculate_field(const int i, jams::Real time) {
  Vec3 field = {0.0, 0.0, 0.0};

    Vec3 spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};

    Vec3 u = {u_axes_(i, 0), u_axes_(i, 1), u_axes_(i, 2)};
    Vec3 v = {v_axes_(i, 0), v_axes_(i, 1), v_axes_(i, 2)};
    Vec3 w = {w_axes_(i, 0), w_axes_(i, 1), w_axes_(i, 2)};

    double Su = jams::dot(spin, u);
    double Sv = jams::dot(spin, v);
    double Sw = jams::dot(spin, w);

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

  return jams::array_cast<jams::Real>(field);
}
