#include <functional>
#include <fstream>
#include "jams/helpers/output.h"
#include "jams/hamiltonian/exchange_functional.h"
#include "jams/core/lattice.h"
#include "jams/core/globals.h"

using namespace std;

ExchangeFunctionalHamiltonian::ExchangeFunctionalHamiltonian(const libconfig::Setting &settings,
    const unsigned int size) : SparseInteractionHamiltonian(settings, size) {
  auto exchange_functional = functional_from_settings(settings);
  radius_cutoff_ = jams::config_required<double>(settings, "r_cutoff");

  cout << "  cutoff radius: " << jams::fmt::decimal << radius_cutoff_ << "\n";

  if (radius_cutoff_ > lattice->max_interaction_radius()) {
    throw std::runtime_error("cutoff radius is larger than the maximum radius which avoids self interaction");
  }

  ofstream of(seedname + "_exchange_functional.tsv");
  output_exchange_functional(of, exchange_functional, radius_cutoff_);

  auto counter = 0;
  vector<Atom> nbrs;
  for (auto i = 0; i < globals::num_spins; ++i) {
    nbrs.clear();
    ::lattice->atom_neighbours(i, radius_cutoff_, nbrs);

    for (const auto& nbr : nbrs) {
      const auto j = nbr.id;
      if (i == j) {
        continue;
      }
      const auto rij = norm(lattice->displacement(i, j));
      this->insert_interaction_scalar(i, j, input_unit_conversion_ * exchange_functional(rij));
      counter++;
    }
  }

  cout << "  total interactions " << jams::fmt::integer << counter << "\n";
  cout << "  average interactions per spin " << jams::fmt::decimal << counter / double(globals::num_spins) << "\n";

  finalize();
}

double ExchangeFunctionalHamiltonian::functional_exp(const double rij, const double J0, const double r0, const double sigma){
  return J0 * exp(-(rij - r0) / sigma);
}

double ExchangeFunctionalHamiltonian::functional_rkky(const double rij, const double J0, const double r0, const double k_F) {
  double kr = 2 * k_F * (rij - r0);
  return - J0 * (kr * cos(kr) - sin(kr)) / pow4(kr);
}

double ExchangeFunctionalHamiltonian::functional_gaussian(const double rij, const double J0, const double r0, const double sigma){
  return J0 * exp(-pow2(rij - r0)) / (2 * pow2(sigma));
  //approximation (gives similar shape to that of BS curve)
}

double ExchangeFunctionalHamiltonian::functional_kaneyoshi(const double rij, const double J0, const double r0, const double sigma){
  return J0 * pow2(rij - r0) * exp(-pow2(rij - r0) / (2 * pow2(sigma)));
}

ExchangeFunctionalHamiltonian::ExchangeFunctional
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
  } else if (functional_name == "kaneyoshi") {
    return bind(functional_kaneyoshi, _1, double(settings["J0"]), double(settings["r0"]), double(settings["sigma"]));
  } else {
    throw runtime_error("unknown exchange functional: " + functional_name);
  }
}

void
ExchangeFunctionalHamiltonian::output_exchange_functional(std::ostream &os,
    const ExchangeFunctionalHamiltonian::ExchangeFunctional& functional, double r_cutoff, double delta_r) {
  os << "radius    exchange" << "\n";
  double r = 0.0;
  while (r < r_cutoff) {
    os << jams::fmt::decimal << r << " " << jams::fmt::sci << functional(r) << "\n";
    r += delta_r;
  }
}


