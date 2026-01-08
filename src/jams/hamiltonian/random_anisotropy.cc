//
// Created by Joe Barker on 2018/05/28.
//
#include <fstream>
#include <random>
#include <vector>

#include <libconfig.h++>
#include <pcg_random.hpp>

#include <jams/core/globals.h>
#include <jams/common.h>
#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/hamiltonian/random_anisotropy.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/helpers/output.h"
#include <jams/helpers/exception.h>

RandomAnisotropyHamiltonian::RandomAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : Hamiltonian(settings, size),
          magnitude_(size, 0.0),
          direction_(size, {0.0, 0.0, 0.0})
{
  // validate settings
  if (settings["magnitude"].getLength() != globals::lattice->num_materials()) {
    throw jams::ConfigException(settings["magnitude"], "magnitude must be specified for every material");
  }

  if (settings.exists("sigma")) {
    if (settings["sigma"].getLength() != globals::lattice->num_materials()) {
      throw jams::ConfigException(settings["sigma"], "sigma must be specified for every material");
    }
  }

  struct RandomAnisotropyProperties {
      jams::Real magnitude;
      jams::Real sigma;
  };

  std::vector<RandomAnisotropyProperties> properties;
  for (auto i = 0; i < settings["magnitude"].getLength(); ++i) {
    properties.push_back({
                             static_cast<jams::Real>(jams::Real(settings["magnitude"][i]) * input_energy_unit_conversion_),
            settings.exists("sigma") ? jams::Real(settings["sigma"][i]) : jams::Real(0)});
  }

  auto seed = jams::config_optional<unsigned>(settings, "seed", jams::instance().random_generator()());
  std::cout << "    seed " << seed << "\n";

  pcg32 generator(seed);
  auto random_unit_vector = std::bind(uniform_random_sphere<jams::Real, pcg32>, generator);
  auto random_normal_number = std::bind(std::normal_distribution<jams::Real>(), generator);

  for (auto i = 0; i < size; ++i) {
    auto type = globals::lattice->lattice_site_material_id(i);
    magnitude_[i] = properties[type].magnitude + properties[type].sigma * random_normal_number();
    direction_[i] = random_unit_vector();
  }

  if (debug_is_enabled() || verbose_is_enabled()) {
    std::ofstream outfile(jams::output::full_path_filename("DEBUG_random_anisotropy.tsv"));
    output_anisotropy_axes(outfile);
  }
}


jams::Real RandomAnisotropyHamiltonian::calculate_energy(const int i, jams::Real time) {
  return -magnitude_[i] * pow2(direction_[i][0] * globals::s(i, 0) + direction_[i][1] * globals::s(i, 1) + direction_[i][2] * globals::s(i, 2));
}


Vec3R RandomAnisotropyHamiltonian::calculate_field(const int i, jams::Real time) {
  Vec3R s_i = array_cast<jams::Real>(Vec3{globals::s(i,0), globals::s(i,1), globals::s(i,2)});
  return magnitude_[i] * dot(direction_[i], s_i) * direction_[i];
}


void RandomAnisotropyHamiltonian::output_anisotropy_axes(std::ofstream &outfile) {
  outfile << std::setw(12) << "index";
  outfile << std::setw(12) << "rx ry rz";
  outfile << std::setw(12) << "ex ey ez";
  outfile << std::setw(12) << "D";
  outfile << "\n";

  for (auto i = 0; i < direction_.size(); ++i) {
    auto r = globals::lattice->lattice_site_position_cart(i);
    outfile << std::setw(12) << i;
    outfile << std::setw(12) << r;
    outfile << std::setw(12) << direction_[i];
    outfile << std::setw(12) << magnitude_[i];
    outfile << "\n";
  }
}