//
// Created by Joe Barker on 2018/05/28.
//
#include <fstream>
#include <random>
#include <vector>

#include <libconfig.h++>
#include <pcg_random.hpp>

#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/random_anisotropy.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/helpers/output.h"

RandomAnisotropyHamiltonian::RandomAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : Hamiltonian(settings, size),
          magnitude_(size, 0.0),
          direction_(size, {0.0, 0.0, 0.0})
{
  using namespace std;

  // validate settings
  if (settings["magnitude"].getLength() != globals::lattice->num_materials()) {
    jams_die("RandomAnisotropyHamiltonian: magnitude must be specified for every material");
  }

  if (settings.exists("sigma")) {
    if (settings["sigma"].getLength() != globals::lattice->num_materials()) {
      jams_die("RandomAnisotropyHamiltonian: sigma must be specified for every material");
    }
  }

  struct RandomAnisotropyProperties {
      double magnitude;
      double sigma;
  };

  vector<RandomAnisotropyProperties> properties;
  for (auto i = 0; i < settings["magnitude"].getLength(); ++i) {
    properties.push_back({
                             double(settings["magnitude"][i]) * input_energy_unit_conversion_,
            settings.exists("sigma") ? double(settings["sigma"][i]) : 0.0});
  }

  auto seed = jams::config_optional<unsigned>(settings, "seed", jams::instance().random_generator()());
  cout << "    seed " << seed << "\n";

  pcg32 generator(seed);
  auto random_unit_vector = bind(uniform_random_sphere<pcg32>, generator);
  auto random_normal_number = bind(normal_distribution<>(), generator);

  for (auto i = 0; i < size; ++i) {
    auto type = globals::lattice->atom_material_id(i);
    magnitude_[i] = properties[type].magnitude + properties[type].sigma * random_normal_number();
    direction_[i] = random_unit_vector();
  }

  if (debug_is_enabled() || verbose_is_enabled()) {
    ofstream outfile(jams::output::full_path_filename("DEBUG_random_anisotropy.tsv"));
    output_anisotropy_axes(outfile);
  }
}

double RandomAnisotropyHamiltonian::calculate_total_energy(double time) {
  double total_energy = 0.0;
  for (auto i = 0; i < energy_.size(); ++i) {
    total_energy += calculate_energy(i, time);
  }
  return total_energy;
}

double RandomAnisotropyHamiltonian::calculate_energy(const int i, double time) {
  return -magnitude_[i] * pow2(direction_[i][0] * globals::s(i, 0) + direction_[i][1] * globals::s(i, 1) + direction_[i][2] * globals::s(i, 2));
}

double RandomAnisotropyHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
                                                                const Vec3 &spin_final, double time) {
  auto e_initial = -magnitude_[i] * pow2(direction_[i][0] * spin_initial[0] + direction_[i][1] * spin_initial[1] + direction_[i][2] * spin_initial[2]);
  auto e_final =   -magnitude_[i] * pow2(direction_[i][0] * spin_final[0] + direction_[i][1] * spin_final[1] + direction_[i][2] * spin_final[2]);

  return e_final - e_initial;
}

void RandomAnisotropyHamiltonian::calculate_energies(double time) {
  for (auto i = 0; i < energy_.size(); ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}

Vec3 RandomAnisotropyHamiltonian::calculate_field(const int i, double time) {
  Vec3 s_i = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
  return magnitude_[i] * dot(direction_[i], s_i) * direction_[i];
}

void RandomAnisotropyHamiltonian::calculate_fields(double time) {
  for (auto i = 0; i < field_.size(0); ++i) {
    auto field = calculate_field(i, time);
    for (auto n = 0; n < 3; ++n) {
      field_(i, n) = field[n];
    }
  }
}

void RandomAnisotropyHamiltonian::output_anisotropy_axes(std::ofstream &outfile) {
  outfile << std::setw(12) << "index";
  outfile << std::setw(12) << "rx ry rz";
  outfile << std::setw(12) << "ex ey ez";
  outfile << std::setw(12) << "D";
  outfile << "\n";

  for (auto i = 0; i < direction_.size(); ++i) {
    auto r = globals::lattice->atom_position(i);
    outfile << std::setw(12) << i;
    outfile << std::setw(12) << r;
    outfile << std::setw(12) << direction_[i];
    outfile << std::setw(12) << magnitude_[i];
    outfile << "\n";
  }
}