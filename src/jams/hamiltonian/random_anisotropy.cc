//
// Created by Joe Barker on 2018/05/28.
//
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
#include <jams/interface/config.h>

RandomAnisotropyHamiltonian::RandomAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : Hamiltonian(settings, size),
          magnitude_(size, 0.0),
          direction_(size, {0.0, 0.0, 0.0})
{
  const auto num_materials = globals::lattice->num_materials();
  const auto magnitudes = jams::read_numeric_sequence_setting<jams::Real>(
      settings["magnitude"], "magnitude", num_materials);
  std::vector<jams::Real> sigmas(num_materials, jams::Real{0});
  if (settings.exists("sigma")) {
    sigmas = jams::read_numeric_sequence_setting<jams::Real>(settings["sigma"], "sigma", num_materials);
  }

  struct RandomAnisotropyProperties {
      jams::Real magnitude;
      jams::Real sigma;
  };

  std::vector<RandomAnisotropyProperties> properties;
  properties.reserve(num_materials);
  for (auto i = 0; i < num_materials; ++i) {
    properties.push_back({
        static_cast<jams::Real>(magnitudes[i] * input_energy_unit_conversion_),
        sigmas[i]});
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
    jams::output::TsvWriter tsv(
        jams::output::hamiltonian_filename(name() + "_random_anisotropy_axes", "tsv"),
        {{"index", "none", jams::output::ColFmt::Integer},
         {"rx", "lattice constants"},
         {"ry", "lattice constants"},
         {"rz", "lattice constants"},
         {"ex", "dimensionless"},
         {"ey", "dimensionless"},
         {"ez", "dimensionless"},
         {"D", "meV"}});
    output_anisotropy_axes(tsv);
  }
}


jams::Real RandomAnisotropyHamiltonian::calculate_energy(const int i, jams::Real time) {
  return calculate_energy_for_spin(i, {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}, time);
}

jams::Real RandomAnisotropyHamiltonian::calculate_energy_for_spin(const int i, const jams::Vec<double, 3>& spin, jams::Real time) {
  const auto s = jams::array_cast<jams::Real>(spin);
  return -magnitude_[i] * pow2(jams::dot(direction_[i], s));
}


jams::Vec<jams::Real, 3> RandomAnisotropyHamiltonian::calculate_field(const int i, jams::Real time) {
  jams::Vec<jams::Real, 3> s_i = jams::array_cast<jams::Real>(jams::Vec<double, 3>{globals::s(i,0), globals::s(i,1), globals::s(i,2)});
  return magnitude_[i] * jams::dot(direction_[i], s_i) * direction_[i];
}


void RandomAnisotropyHamiltonian::output_anisotropy_axes(jams::output::TsvWriter& tsv) {
  for (auto i = 0; i < direction_.size(); ++i) {
    auto r = globals::lattice->lattice_site_position_cart(i);
    tsv.write_row_values(
        i,
        r[0],
        r[1],
        r[2],
        direction_[i][0],
        direction_[i][1],
        direction_[i][2],
        magnitude_[i]);
  }
}
