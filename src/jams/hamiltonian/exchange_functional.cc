#include <functional>
#include <fstream>
#include "jams/helpers/output.h"
#include "jams/hamiltonian/exchange_functional.h"
#include "jams/core/lattice.h"
#include "jams/core/globals.h"

#include <jams/lattice/interaction_neartree.h>

using namespace std;

ExchangeFunctionalHamiltonian::ExchangeFunctionalHamiltonian(const libconfig::Setting &settings,
    const unsigned int size) : SparseInteractionHamiltonian(settings, size) {
  auto exchange_functional = functional_from_settings(settings);
  radius_cutoff_ = jams::config_required<double>(settings, "r_cutoff");

  cout << "  cutoff radius: " << jams::fmt::decimal << radius_cutoff_ << "\n";
  cout << "  max cutoff radius: " << lattice->max_interaction_radius() << "\n";

  if (radius_cutoff_ > lattice->max_interaction_radius()) {
    throw std::runtime_error("cutoff radius is larger than the maximum radius which avoids self interaction");
  }

  ofstream of(jams::output::full_path_filename("exchange_functional.tsv"));
  output_exchange_functional(of, exchange_functional, radius_cutoff_);

  jams::InteractionNearTree neartree(lattice->get_supercell().a(), lattice->get_supercell().b(), lattice->get_supercell().c(), lattice->periodic_boundaries(), radius_cutoff_, jams::defaults::lattice_tolerance);
  neartree.insert_sites(lattice->atom_cartesian_positions());

  auto counter = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r_i = lattice->atom_position(i);
    const auto nbrs = neartree.neighbours(r_i, radius_cutoff_);

    for (const auto& nbr : nbrs) {
      const auto j = nbr.second;
      if (i == j) {
        continue;
      }
      const auto rij = norm(::lattice->displacement(i, j));
      this->insert_interaction_scalar(i, j, input_unit_conversion_ * exchange_functional(rij));
      counter++;
    }
  }

  cout << "  total interactions " << jams::fmt::integer << counter << "\n";
  cout << "  average interactions per spin " << jams::fmt::decimal << counter / double(globals::num_spins) << "\n";

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

  const string functional_name = lowercase(jams::config_required<string>(settings, "functional"));
  cout << "  exchange functional: " << functional_name << "\n";

  if (functional_name == "rkky") {
    return bind(functional_rkky, _1, double(settings["J0"]), double(settings["r0"]), double(settings["k_F"]));
  } else if (functional_name == "exponential") {
    return bind(functional_exp, _1, double(settings["J0"]), double(settings["r0"]), double(settings["sigma"]));
  } else if (functional_name == "gaussian") {
    return bind(functional_gaussian, _1, double(settings["J0"]), double(settings["r0"]), double(settings["sigma"]));
  } else if (functional_name == "gaussian_multi") {
    return bind(functional_gaussian_multi, _1,
                double(settings["J0"]), double(settings["r0"]), double(settings["sigma0"]),
                double(settings["J1"]), double(settings["r1"]), double(settings["sigma1"]),
                double(settings["J2"]), double(settings["r2"]), double(settings["sigma2"]));
  } else if (functional_name == "kaneyoshi") {
    return bind(functional_kaneyoshi, _1, double(settings["J0"]), double(settings["r0"]), double(settings["sigma"]));
  } else if (functional_name == "step") {
    return bind(functional_step, _1, double(settings["J0"]), double(settings["r_cutoff"]));
  } else {
    throw runtime_error("unknown exchange functional: " + functional_name);
  }
}

void
ExchangeFunctionalHamiltonian::output_exchange_functional(std::ostream &os,
                                                          const ExchangeFunctionalHamiltonian::ExchangeFunctionalType& functional, double r_cutoff, double delta_r) {
  os << "radius    exchange" << "\n";
  double r = 0.0;
  while (r < r_cutoff) {
    os << jams::fmt::decimal << r << " " << jams::fmt::sci << functional(r) << "\n";
    r += delta_r;
  }
}


