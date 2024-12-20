#include <functional>
#include <fstream>
#include "jams/helpers/output.h"
#include "jams/hamiltonian/exchange_functional.h"
#include "jams/core/lattice.h"
#include "jams/core/globals.h"

#include <jams/lattice/interaction_neartree.h>

ExchangeFunctionalHamiltonian::ExchangeFunctionalHamiltonian(const libconfig::Setting &settings,
    const unsigned int size) : SparseInteractionHamiltonian(settings, size) {

  auto exchange_functional = functional_from_settings(settings);

  radius_cutoff_ = input_distance_unit_conversion_ * jams::config_required<double>(settings, "r_cutoff");


  std::cout << "  cutoff radius: " << jams::fmt::decimal << radius_cutoff_ << " (latt_const)\n";
  std::cout << "  max cutoff radius: " << globals::lattice->max_interaction_radius() << " (latt_const)\n";

  if (radius_cutoff_ > globals::lattice->max_interaction_radius()) {
    throw std::runtime_error("cutoff radius is larger than the maximum radius which avoids self interaction");
  }

  std::ofstream of(jams::output::full_path_filename("exchange_functional.tsv"));
  output_exchange_functional(of, exchange_functional, radius_cutoff_);

  jams::InteractionNearTree neartree(globals::lattice->get_supercell().a1(),
                                     globals::lattice->get_supercell().a2(),
                                     globals::lattice->get_supercell().a3(), globals::lattice->periodic_boundaries(), radius_cutoff_, jams::defaults::lattice_tolerance);
  neartree.insert_sites(globals::lattice->lattice_site_positions_cart());

  double total_abs_exchange = 0.0;

  auto counter = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r_i = globals::lattice->lattice_site_position_cart(i);
    const auto nbrs = neartree.neighbours(r_i, radius_cutoff_);

    for (const auto& nbr : nbrs) {
      const auto j = nbr.second;
      if (i == j) {
        continue;
      }
      const auto rij = norm(::globals::lattice->displacement(i, j));
      const auto Jij = exchange_functional(rij);
      this->insert_interaction_scalar(i, j, Jij);
      counter++;
      total_abs_exchange += std::abs(Jij);
    }
  }

  std::cout << "  total interactions " << jams::fmt::integer << counter << "\n";
  std::cout << "  average interactions per spin " << jams::fmt::decimal << counter / double(globals::num_spins) << "\n";
  std::cout << "  average abs exchange energy per spin (meV)" << jams::fmt::decimal << total_abs_exchange / double(globals::num_spins) << "\n";

  finalize(jams::SparseMatrixSymmetryCheck::Symmetric);
}


double ExchangeFunctionalHamiltonian::functional_step(double rij, double J0, double r_cut) {
  if (rij < r_cut) {
    return J0;
  }
  return 0.0;
}

double ExchangeFunctionalHamiltonian::functional_exp(const double rij, const double J0, const double r0, const double sigma){
  return J0 * exp(-(rij - r0) / sigma);
}

double ExchangeFunctionalHamiltonian::functional_rkky(const double rij, const double J0, const double r0, const double k_F) {
  double kr = 2 * k_F * (rij - r0);
  return - J0 * (kr * cos(kr) - sin(kr)) / pow4(kr);
}

double ExchangeFunctionalHamiltonian::functional_gaussian(const double rij, const double J0, const double r0, const double sigma){
  return J0 * exp(-pow2(rij - r0)/(2 * pow2(sigma)));
}

double ExchangeFunctionalHamiltonian::functional_gaussian_multi(const double rij, const double J0, const double r0, const double sigma0, const double J1, const double r1, const double sigma1, const double J2, const double r2, const double sigma2) {
  return functional_gaussian(rij, J0, r0, sigma0) + functional_gaussian(rij, J1, r1, sigma1) + functional_gaussian(rij, J2, r2, sigma2);
}

double ExchangeFunctionalHamiltonian::functional_kaneyoshi(const double rij, const double J0, const double r0, const double sigma){
  return J0 * pow2(rij - r0) * exp(-pow2(rij - r0) / (2 * pow2(sigma)));
}

ExchangeFunctionalHamiltonian::ExchangeFunctionalType
ExchangeFunctionalHamiltonian::functional_from_settings(const libconfig::Setting &settings) {
  using namespace std::placeholders;

  const std::string functional_name = lowercase(jams::config_required<std::string>(settings, "functional"));
  std::cout << "  exchange functional: " << functional_name << "\n";

  if (functional_name == "rkky") {
    return std::bind(functional_rkky, _1,
                input_energy_unit_conversion_ * double(settings["J0"]),
                input_distance_unit_conversion_ * double(settings["r0"]),
                double(settings["k_F"]));
  } else if (functional_name == "exponential") {
    return std::bind(functional_exp, _1,
                input_energy_unit_conversion_ * double(settings["J0"]),
                input_distance_unit_conversion_ * double(settings["r0"]),
                input_distance_unit_conversion_ * double(settings["sigma"]));
  } else if (functional_name == "gaussian") {
    return std::bind(functional_gaussian, _1,
                input_energy_unit_conversion_ * double(settings["J0"]),
                input_distance_unit_conversion_ * double(settings["r0"]),
                input_distance_unit_conversion_ * double(settings["sigma"]));
  } else if (functional_name == "gaussian_multi") {
    return std::bind(functional_gaussian_multi, _1,
                input_energy_unit_conversion_ * double(settings["J0"]),
                input_distance_unit_conversion_ * double(settings["r0"]),
                input_distance_unit_conversion_ * double(settings["sigma0"]),
                input_energy_unit_conversion_ * double(settings["J1"]),
                input_distance_unit_conversion_ * double(settings["r1"]),
                input_distance_unit_conversion_ * double(settings["sigma1"]),
                input_energy_unit_conversion_ * double(settings["J2"]),
                input_distance_unit_conversion_ * double(settings["r2"]),
                input_distance_unit_conversion_ * double(settings["sigma2"]));
  } else if (functional_name == "kaneyoshi") {
    return std::bind(functional_kaneyoshi, _1,
                input_energy_unit_conversion_ * double(settings["J0"]),
                input_distance_unit_conversion_ * double(settings["r0"]),
                input_distance_unit_conversion_ * double(settings["sigma"]));
  } else if (functional_name == "step") {
    return std::bind(functional_step, _1,
                input_energy_unit_conversion_ * double(settings["J0"]),
                input_distance_unit_conversion_ * double(settings["r_cutoff"]));
  } else {
    throw std::runtime_error("unknown exchange functional: " + functional_name);
  }
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


