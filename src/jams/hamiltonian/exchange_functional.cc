#include <functional>
#include <fstream>
#include "jams/helpers/output.h"
#include "jams/hamiltonian/exchange_functional.h"
#include "jams/core/lattice.h"
#include "jams/core/globals.h"

#include <jams/lattice/interaction_neartree.h>


ExchangeFunctionalHamiltonian::ExchangeFunctionalHamiltonian(const libconfig::Setting &settings,
    const unsigned int size) : SparseInteractionHamiltonian(settings, size) {


  std::map<std::pair<std::string, std::string>, std::pair<double, ExchangeFunctionalType>> exchange_functional_map;


  double max_cutoff_radius = 0.0;
  for (auto n = 0; n < settings["interactions"].getLength(); ++n) {
    auto type_i = std::string(settings["interactions"][n][0]);
    auto type_j = std::string(settings["interactions"][n][1]);
    auto functional_name = std::string(settings["interactions"][n][2]);
    auto r_cutoff = input_distance_unit_conversion_ * double(settings["interactions"][n][3]);

    // Check that this pair (in either order) has not been specified before
    const auto key_ij = std::make_pair(type_i, type_j);
    const auto key_ji = std::make_pair(type_j, type_i);

    if (exchange_functional_map.find(key_ij) != exchange_functional_map.end() ||
        exchange_functional_map.find(key_ji) != exchange_functional_map.end()) {
      throw std::runtime_error(
          "Interaction between types \"" + type_i + "\" and \"" + type_j +
          "\" is defined more than once (order does not matter).");
        }

    if (r_cutoff > globals::lattice->max_interaction_radius()) {
      throw std::runtime_error(
          "cutoff radius " + std::to_string(r_cutoff) +
          " is larger than the maximum cutoff radius " +
          std::to_string(globals::lattice->max_interaction_radius()));
    }

    if (r_cutoff > max_cutoff_radius) {
      max_cutoff_radius = r_cutoff;
    }

    std::vector<double> params;
    for (auto k = 4; k < settings["interactions"][n].getLength(); ++k) {
      params.push_back(settings["interactions"][n][k]);
    }

    auto exchange_functional = functional_from_params(functional_name, params);

    // Now safe to insert
    exchange_functional_map[key_ij] = {r_cutoff, exchange_functional};

    if (type_i != type_j) {
      exchange_functional_map[key_ji] = {r_cutoff, exchange_functional};
    }
  }

  auto output_functionals = jams::config_optional<bool>(settings, "output_functionals", false);

  if (output_functionals) {
    for (const auto& [type, functional] : exchange_functional_map) {
      std::ofstream functional_file(jams::output::full_path_filename("exchange_functional_" + type.first + "_" + type.second + ".tsv"));
      output_exchange_functional(functional_file, functional.second, functional.first, 0.01);
    }
  }



  jams::InteractionNearTree neartree(globals::lattice->get_supercell().a1(),
                                     globals::lattice->get_supercell().a2(),
                                     globals::lattice->get_supercell().a3(), globals::lattice->periodic_boundaries(), max_cutoff_radius, jams::defaults::lattice_tolerance);
  neartree.insert_sites(globals::lattice->lattice_site_positions_cart());

  auto cartesian_positions = globals::lattice->lattice_site_positions_cart();

  auto counter = 0;
  std::vector<int> seen_stamp(globals::num_spins, -1);

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto type_i = globals::lattice->lattice_site_material_name(i);

    auto r_i = globals::lattice->lattice_site_position_cart(i);
    const auto nbrs = neartree.neighbours(r_i, max_cutoff_radius);

    for (const auto& [rij, j] : nbrs) {
      // Only process ij, ji is inserted at the same time. Also disallow self interaction.
      if (j <= i) {
        continue;
      }
      auto type_j = globals::lattice->lattice_site_material_name(j);
      auto& [r_cutoff, functional] = exchange_functional_map[{type_i, type_j}];

      const auto r = norm(rij);

      if (less_than_approx_equal(r, r_cutoff, jams::defaults::lattice_tolerance)) {
        // don't allow self interaction
        if (seen_stamp[j] == i) {
          throw jams::SanityException("multiple interactions between spins ", i, " and ", j);
        }
        seen_stamp[j] = i;

        // We insert ij and ji at the same time because tiny floating point differences of
        // functional(rij) vs functional(rji) can lead to the sparse matrix being a tiny bit non-symmetric.
        // This probably makes no difference to results, but means that our symmetry check for the matrix
        // will fail because we don't do floating point equality checks, but check that values for ij and ji
        // are identical.
        auto Jij = functional(r);
        this->insert_interaction_scalar(i, j, Jij);
        this->insert_interaction_scalar(j, i, Jij);
        counter++;
      }
    }
  }



  std::cout << "  total interactions " << jams::fmt::integer << counter << "\n";
  std::cout << "  average interactions per spin " << jams::fmt::decimal << counter / double(globals::num_spins) << "\n";

  finalize(jams::SparseMatrixSymmetryCheck::Symmetric);
}


double ExchangeFunctionalHamiltonian::functional_step(double rij, double J0, double r_cut) {
  if (less_than_approx_equal(rij,  r_cut, jams::defaults::lattice_tolerance)) {
    return J0;
  }
  return 0.0;
}

double ExchangeFunctionalHamiltonian::functional_exp(double rij, double J0, double r0, double sigma){
  return J0 * exp(-(rij - r0) / sigma);
}

double ExchangeFunctionalHamiltonian::functional_rkky(double rij, double J0, double r0, double k_F) {
  double kr = 2 * k_F * (rij - r0);
  return - J0 * (kr * cos(kr) - sin(kr)) / pow4(kr);
}

double ExchangeFunctionalHamiltonian::functional_gaussian(double rij, double J0, double r0, double sigma){
  return J0 * exp(-pow2(rij - r0)/(2 * pow2(sigma)));
}

double ExchangeFunctionalHamiltonian::functional_gaussian_multi(double rij, double J0, double r0, double sigma0, double J1, double r1, double sigma1, double J2, double r2, double sigma2) {
  return functional_gaussian(rij, J0, r0, sigma0) + functional_gaussian(rij, J1, r1, sigma1) + functional_gaussian(rij, J2, r2, sigma2);
}

double ExchangeFunctionalHamiltonian::functional_kaneyoshi(double rij, double J0, double r0, double sigma){
  return J0 * pow2(rij - r0) * exp(-pow2(rij - r0) / (2 * pow2(sigma)));
}


ExchangeFunctionalHamiltonian::ExchangeFunctionalType
ExchangeFunctionalHamiltonian::functional_from_params(const std::string& name, const std::vector<double>& params) {
  using namespace std::placeholders;

  std::cout << "  exchange functional: " << name << "\n";

  if (name == "rkky") {
    return std::bind(functional_rkky, _1,
                input_energy_unit_conversion_   * params[0],  // J0
                input_distance_unit_conversion_ * params[1],  // r0
                                                  params[2]); // k_F
  }
  if (name == "exponential") {
    return std::bind(functional_exp, _1,
                     input_energy_unit_conversion_   * params[0],  // J0
                     input_distance_unit_conversion_ * params[1],  // r0
                     input_distance_unit_conversion_ * params[2]); // sigma
  }
  if (name == "gaussian") {
    return std::bind(functional_gaussian, _1,
                     input_energy_unit_conversion_   * params[0],  // J0
                     input_distance_unit_conversion_ * params[1],  // r0
                     input_distance_unit_conversion_ * params[2]); // sigma
  }
  if (name == "gaussian_multi") {
    return std::bind(functional_gaussian_multi, _1,
                     input_energy_unit_conversion_   * params[0],  // J0
                     input_distance_unit_conversion_ * params[1],  // r0
                     input_distance_unit_conversion_ * params[2],  // sigma0
                     input_energy_unit_conversion_   * params[3],  // J1
                     input_distance_unit_conversion_ * params[4],  // r1
                     input_distance_unit_conversion_ * params[5],  // sigma1
                     input_energy_unit_conversion_   * params[6],  // J2
                     input_distance_unit_conversion_ * params[7],  // r2
                     input_distance_unit_conversion_ * params[8]); // sigma2
  }
  if (name == "kaneyoshi") {
    return std::bind(functional_kaneyoshi, _1,
                     input_energy_unit_conversion_   * params[0],  // J0
                     input_distance_unit_conversion_ * params[1],  // r0
                     input_distance_unit_conversion_ * params[2]); // sigma
  }
  if (name == "step") {
    return std::bind(functional_step, _1,
                     input_energy_unit_conversion_   * params[0],  // J0
                     input_distance_unit_conversion_ * params[1]); // r_cutoff
  }
  throw std::runtime_error("unknown exchange functional: " + name);
}

void
ExchangeFunctionalHamiltonian::output_exchange_functional(std::ostream &os,
                                                          const ExchangeFunctionalHamiltonian::ExchangeFunctionalType& functional, double r_cutoff, double delta_r) {
  os << "radius_nm  exchange_meV\n";
  double r = 0.0;
  while (r < r_cutoff) {
    os << jams::fmt::decimal << r * ::globals::lattice->parameter() * 1e9 << " " << jams::fmt::sci << functional(r) << "\n";
    r += delta_r;
  }
}


