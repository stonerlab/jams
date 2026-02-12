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
        exchange_functional_map.find(key_ji) != exchange_functional_map.end())
    {
      throw std::runtime_error(
          "Interaction between types \"" + type_i + "\" and \"" + type_j +
          "\" is defined more than once (order does not matter).");
    }

    if (r_cutoff > globals::lattice->max_interaction_radius())
    {
      throw std::runtime_error(
          "cutoff radius " + std::to_string(r_cutoff) +
          " is larger than the maximum cutoff radius " +
          std::to_string(globals::lattice->max_interaction_radius()));
    }

    if (r_cutoff > max_cutoff_radius)
    {
      max_cutoff_radius = r_cutoff;
    }

    std::vector<double> params;
    for (auto k = 4; k < settings["interactions"][n].getLength(); ++k) {
      if (settings["interactions"][n][k].getType() == libconfig::Setting::TypeList || settings["interactions"][n][k].getType() == libconfig::Setting::TypeArray) {
        for (auto l = 0; l < settings["interactions"][n][k].getLength(); ++l) {
          params.push_back(settings["interactions"][n][k][l]);
        }
      } else {
        params.push_back(settings["interactions"][n][k]);
      }
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
      output_exchange_functional(functional_file, functional.second, functional.first);
    }
  }



  jams::InteractionNearTree neartree(globals::lattice->get_supercell().a1(),
                                     globals::lattice->get_supercell().a2(),
                                     globals::lattice->get_supercell().a3(),
                                     globals::lattice->periodic_boundaries(),
                                     max_cutoff_radius, jams::defaults::lattice_tolerance);

  auto cartesian_positions = globals::lattice->lattice_site_positions_cart();
  neartree.insert_sites(globals::lattice->lattice_site_positions_cart());

  auto counter = 0;
  std::vector<int> seen_stamp(globals::num_spins, -1);

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto type_i = globals::lattice->lattice_site_material_name(i);

    auto r_i = cartesian_positions[i];
    const auto nbrs = neartree.neighbours(r_i, max_cutoff_radius);

    for (const auto& [r_j, j] : nbrs) {
      // Only process ij, ji is inserted at the same time. Also disallow self interaction.
      if (j <= i) {
        continue;
      }

      auto r_ij = r_j - r_i;
      auto type_j = globals::lattice->lattice_site_material_name(j);

      using Key = std::pair<std::string,std::string>; // or whatever your key types are
      Key k{type_i, type_j};

      auto it = exchange_functional_map.find(k);
      if (it == exchange_functional_map.end()) {
        // optionally try symmetric lookup
        it = exchange_functional_map.find(Key{type_j, type_i});
      }
      if (it == exchange_functional_map.end()) {
        continue;
      }

      auto& [r_cutoff, functional] = it->second;
      const auto r = norm(r_ij);

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
        auto Jij = functional(r_ij);
        this->insert_interaction_scalar(i, j, Jij);
        this->insert_interaction_scalar(j, i, Jij);
        counter+=2;
      }
    }
  }



  std::cout << "  total interactions " << jams::fmt::integer << counter << "\n";
  std::cout << "  average interactions per spin " << jams::fmt::decimal << counter / double(globals::num_spins) << "\n";

  finalize(jams::SparseMatrixSymmetryCheck::Symmetric);
}


double ExchangeFunctionalHamiltonian::functional_step(Vec3 rij, double J0, double r_cut) {
  double r = norm(rij);
  if (less_than_approx_equal(r,  r_cut, jams::defaults::lattice_tolerance)) {
    return J0;
  }
  return 0.0;
}

double ExchangeFunctionalHamiltonian::functional_exp(Vec3 rij, double J0, double r0, double sigma){
  double r = norm(rij);
  return J0 * exp(-(r - r0) / sigma);
}

double ExchangeFunctionalHamiltonian::functional_rkky(Vec3 rij, double J0, double r0, double k_F) {
  double r = norm(rij);
  double kr = 2 * k_F * (r - r0);
  return - J0 * (kr * cos(kr) - sin(kr)) / pow4(kr);
}

double ExchangeFunctionalHamiltonian::functional_gaussian(Vec3 rij, double J0, double r0, double sigma){
  double r = norm(rij);
  return J0 * exp(-pow2(r - r0)/(2 * pow2(sigma)));
}

double ExchangeFunctionalHamiltonian::functional_gaussian_multi(Vec3 rij, double J0, double r0, double sigma0, double J1, double r1, double sigma1, double J2, double r2, double sigma2) {
  return functional_gaussian(rij, J0, r0, sigma0) + functional_gaussian(rij, J1, r1, sigma1) + functional_gaussian(rij, J2, r2, sigma2);
}

double ExchangeFunctionalHamiltonian::functional_kaneyoshi(Vec3 rij, double J0, double r0, double sigma){
  double r = norm(rij);
  return J0 * pow2(r - r0) * exp(-pow2(r - r0) / (2 * pow2(sigma)));
}


double ExchangeFunctionalHamiltonian::functional_c3z(Vec3 rij,
  Vec3 qs1,
  Vec3 qc1,
  double J0,
  double J1s,
  double J1c,
  double d0,
  double l0,
  double l1s,
  double l1c,
  double rstar) {

  double r = norm(rij);
  Vec3 r_para{rij[0], rij[1], 0.0};

  double term_0 = J0 * exp(-std::abs(r - d0)/l0);

  double sin_sum = 0.0;
  Vec3 qs[3] = {qs1, rotation_matrix_z(2*kPi / 3.0) * qs1, rotation_matrix_z(4*kPi / 3.0) * qs1};
  for (const auto & q : qs) {
    sin_sum += sin(dot(q, r_para));
  }
  const double term_s1 = J1s * exp(-std::abs(r - rstar) / l1s) * sin_sum;;

  double cos_sum = 0.0;
  Vec3 qc[3] = {qc1, rotation_matrix_z(2*kPi / 3.0) * qc1, rotation_matrix_z(4*kPi / 3.0) * qc1};
  for (const auto & q : qc) {
    cos_sum += cos(dot(q, r_para));
  }
  const double term_c1 = J1c * exp(-std::abs(r - rstar) / l1c) * cos_sum;

  return (term_0 + term_s1 + term_c1);
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
  if (name == "c3z") {
    return std::bind(functional_c3z, _1,
      (1.0 / input_distance_unit_conversion_) * Vec3{params[0], params[1], params[2]}, // qs
      (1.0 / input_distance_unit_conversion_) * Vec3{params[3], params[4], params[5]}, // qc
      input_energy_unit_conversion_ * params[6], // J0
      input_energy_unit_conversion_ * params[7], // J1s
      input_energy_unit_conversion_ * params[8], // J1c
      input_distance_unit_conversion_ * params[9], // d0
      input_distance_unit_conversion_ * params[10], // l0
      input_distance_unit_conversion_ * params[11], // l1s
      input_distance_unit_conversion_ * params[12], // l1c
      input_distance_unit_conversion_ * params[13]); // rstar
  }
  if (name == "step") {
    return std::bind(functional_step, _1,
                     input_energy_unit_conversion_   * params[0],  // J0
                     input_distance_unit_conversion_ * params[1]); // r_cutoff
  }
  throw std::runtime_error("unknown exchange functional: " + name);
}

void
ExchangeFunctionalHamiltonian::output_exchange_functional(
    std::ostream &os,
    const ExchangeFunctionalHamiltonian::ExchangeFunctionalType &functional,
    double r_cutoff)
{
  const double a = ::globals::lattice->parameter(); // metres
  const auto delta_r = r_cutoff / 100.0;

  const int n = static_cast<int>(std::ceil(r_cutoff / delta_r));
  os << "x_nm  y_nm  z_nm  exchange_meV\n";

  for (int ix = -n; ix <= n; ++ix) {
    const double x = ix * delta_r;

    for (int iy = -n; iy <= n; ++iy) {
      const double y = iy * delta_r;

      for (int iz = -n; iz <= n; ++iz) {
        const double z = iz * delta_r;

        Vec3 rij = {x, y, z};

        // const double r = norm(rij);
        // if (r > r_cutoff) {
          // continue;
        // }

        os << jams::fmt::decimal
           << x * a * 1e9 << " "
           << y * a * 1e9 << " "
           << z * a * 1e9 << " "
           << jams::fmt::sci
           << functional(rij)
           << "\n";
      }
    }
  }
}

