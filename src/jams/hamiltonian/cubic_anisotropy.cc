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

namespace {
    struct AnisotropySetting_cube {
        unsigned order;
        double energy;
        Vec3 axis1;
        Vec3 axis2;
        Vec3 axis3;
    };

    unsigned cubic_anisotropy_order_from_name(const string name) {
      if (name == "K1") return 1; // This is outputted correctly
      if (name == "K2") return 2;
      throw runtime_error("Unsupported anisotropy: " + name);
    }

    AnisotropySetting_cube read_anisotropy_setting_cube(Setting &setting) {
      if (setting.isList()) {
        Vec3 axis1 = {setting[1][0], setting[1][1], setting[1][2]};
        Vec3 axis2 = {setting[2][0], setting[2][1], setting[2][2]};
        Vec3 axis3 = {setting[3][0], setting[3][1], setting[3][2]};
        return AnisotropySetting_cube{cubic_anisotropy_order_from_name(setting.getParent().getName()), setting[0],
                                      normalize(axis1), normalize(axis2), normalize(axis3)};
      }
      if (setting.isScalar()) {
        return AnisotropySetting_cube{cubic_anisotropy_order_from_name(setting.getParent().getName()), setting,
                                      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
      }
      throw runtime_error("Incorrectly formatted cubic_anisotropy anisotropy setting");
    }

    vector<vector<AnisotropySetting_cube>> read_all_cubic_anisotropy_settings(const Setting &settings) {
      vector<vector<AnisotropySetting_cube>> cubic_anisotropies(lattice->num_materials());
      auto anisotropy_cubic_orders = {"K1", "K2"};
      for (const auto name : anisotropy_cubic_orders) {
        if (settings.exists(name)) {
          if (settings[name].getLength() < lattice->num_materials()) {
            throw runtime_error("CubicHamiltonian: " + string(name) + "  must be specified for every material");
          }

          if (settings[name].getLength() > lattice->num_materials()) {
            throw runtime_error("CubicHamiltonian: " + string(name) + "  is specified for too many materials");
          }

          for (auto i = 0; i < settings[name].getLength(); ++i) {
            cubic_anisotropies[i].push_back(read_anisotropy_setting_cube(settings[name][i]));
          }
        }
      }
      // the array indicies are (type, power)
      return cubic_anisotropies;
    }
}

CubicHamiltonian::CubicHamiltonian(const Setting &settings, const unsigned int num_spins)
    : Hamiltonian(settings, num_spins) {

  auto cubic_anisotropies = read_all_cubic_anisotropy_settings(settings);

  std::cout << " material | axis 1 | axis 2 | axis 3 | order | energy" << "\n";
  for (auto type = 0; type < lattice->num_materials(); ++type) {
    std::cout << "  " << lattice->material_name(type) << ": ";
    for (const auto& ani : cubic_anisotropies[type]) {
      std::cout << "   | [" << ani.axis1 << "] | [" << ani.axis2 << "] | [" << ani.axis3 << "] | " << ani.order << " | " << ani.energy << "\n";
    }
  }

  num_coefficients_ = cubic_anisotropies[0].size();

  order_.resize(num_spins, cubic_anisotropies[0].size());
  axis1_.resize(num_spins, cubic_anisotropies[0].size());
  axis2_.resize(num_spins, cubic_anisotropies[0].size());
  axis3_.resize(num_spins, cubic_anisotropies[0].size());
  magnitude_.resize(num_spins, cubic_anisotropies[0].size());

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto type = lattice->atom_material_id(i);
    for (auto j = 0; j < cubic_anisotropies[type].size(); ++j) {
      order_(i, j) = cubic_anisotropies[type][j].order;
      magnitude_(i, j) = cubic_anisotropies[type][j].energy * input_energy_unit_conversion_;
      axis1_(i, j) = cubic_anisotropies[type][j].axis1;
      axis2_(i, j) = cubic_anisotropies[type][j].axis2;
      axis3_(i, j) = cubic_anisotropies[type][j].axis3;
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
  using namespace globals;
  double energy = 0.0;

  for (auto n = 0; n < num_coefficients_; ++n) {
    Vec3 spin = {s(i, 0), s(i, 1), s(i, 2)};

    if(order_(i, n) == 1) {
      energy += -magnitude_(i,n) * (dot_squared(axis1_(i, n), spin) *
                                    dot_squared(axis2_(i, n), spin)
                                    + dot_squared(axis2_(i, n), spin) * dot_squared(
          axis3_(i, n), spin)
                                    + dot_squared(axis3_(i, n), spin) * dot_squared(
          axis1_(i, n), spin) );
    }

    if(order_(i, n) == 2){
      energy += -magnitude_(i,n) * (dot_squared(axis1_(i, n), spin) *
                                    dot_squared(axis2_(i, n), spin) *
                                    dot_squared(axis3_(i, n), spin) );
    }
  }

  return energy;
}

double CubicHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                     const Vec3 &spin_final, double time) {
  double e_initial = 0.0;
  double e_final = 0.0;

  for (auto n = 0; n < num_coefficients_; ++n) {
    if(order_(i, n) == 1) {
      e_initial += -magnitude_(i,n) * (
                                          dot_squared(axis1_(i, n),
                                                      spin_initial) *
                                          dot_squared(axis2_(i, n),
                                                          spin_initial)
                                          + dot_squared(axis2_(i, n), spin_initial) *
            dot_squared(axis3_(i, n), spin_initial)
          + dot_squared(axis3_(i, n), spin_initial) *
            dot_squared(axis1_(i, n), spin_initial) );

      e_final += -magnitude_(i,n) * (dot_squared(axis1_(i, n), spin_final) *
                                     dot_squared(axis2_(i, n), spin_final)
                                     + dot_squared(axis2_(i, n), spin_final) *
                                       dot_squared(
                                                                                    axis3_(
                                                                                        i,
                                                                                        n),
                                                                                    spin_final)
                                     + dot_squared(axis3_(i, n), spin_final) *
                                       dot_squared(
                                                                                                                           axis1_(
                                                                                                                               i,
                                                                                                                               n),
                                                                                                                           spin_final) );
    }

    if(order_(i, n) == 2) {
      e_initial += -magnitude_(i,n) * (dot_squared(axis1_(i, n), spin_initial) *
                                       dot_squared(axis2_(i, n), spin_initial) *
                                       dot_squared(axis3_(i, n), spin_initial) );
      e_final += -magnitude_(i,n) * (dot_squared(axis1_(i, n), spin_final) *
                                     dot_squared(axis2_(i, n), spin_final) *
                                     dot_squared(axis3_(i, n), spin_final) );
    }
  }

  return e_final - e_initial;
}

void CubicHamiltonian::calculate_energies(double time) {
  for (int i = 0; i < energy_.size(); ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}

Vec3 CubicHamiltonian::calculate_field(const int i, double time) {
  using namespace globals;
  Vec3 field = {0.0, 0.0, 0.0};

  for (auto n = 0; n < num_coefficients_; ++n) {
    Vec3 spin = {s(i, 0), s(i, 1), s(i, 2)};
    auto pre = 2.0 * magnitude_(i,n);

    if (order_(i, n) == 1) {
      for (auto j = 0; j < 3; ++j) {
        field[j] += pre * (
            axis1_(i,n)[j] * dot(axis1_(i, n), spin) * (dot_squared(
                axis2_(i, n), spin) +
                                                        dot_squared(axis3_(i, n), spin))
            + axis2_(i,n)[j] * dot(axis2_(i, n), spin) * (dot_squared(
                axis3_(i, n), spin) +
                                                          dot_squared(axis1_(i, n), spin))
            + axis3_(i,n)[j] * dot(axis3_(i, n), spin) * (dot_squared(
                axis1_(i, n), spin) +
                                                          dot_squared(axis2_(i, n), spin)) );
      }
    }

    if (order_(i, n) == 2) {
      for (auto j = 0; j < 3; ++j) {
        field[j] += pre * (
            axis1_(i,n)[j]  * dot(axis1_(i, n), spin) * (dot_squared(
                axis2_(i, n), spin) *
                                                         dot_squared(axis3_(i, n), spin))
            + axis2_(i,n)[j]  * dot(axis2_(i, n), spin) * (dot_squared(
                axis3_(i, n), spin) *
                                                           dot_squared(axis1_(i, n), spin))
            + axis3_(i,n)[j]  * dot(axis3_(i, n), spin) * (dot_squared(
                axis1_(i, n), spin) *
                                                           dot_squared(axis2_(i, n), spin)) );
      }
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