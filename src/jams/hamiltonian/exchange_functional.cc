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
  cout << "  max cutoff radius: " << lattice->max_interaction_radius() << "\n";

  if (radius_cutoff_ > lattice->max_interaction_radius()) {
    throw std::runtime_error("cutoff radius is larger than the maximum radius which avoids self interaction");
  }

  ofstream of(seedname + "_exchange_functional.tsv");
  output_exchange_functional(of, exchange_functional, radius_cutoff_);

  string name3 = seedname + "_spectrum_crystal_limit.tsv";
  ofstream outfile3(name3.c_str());
  outfile3.setf(std::ios::right);

  // header for crystal-limit spectrum file
  outfile3 << std::setw(20) << "kx" << "\t";//(\t=tab)
  outfile3 << std::setw(20) << "ky" << "\t";//(\t=tab)
  outfile3 << std::setw(20) << "kz" << "\t";//(\t=tab)
  outfile3 << std::setw(20) << "Re:E(k)" << "\t";
  outfile3 << std::setw(20) << "Im:E(k)" << "\n";

  auto counter = 0;
  auto counter2 = 0;
  vector<Atom> nbrs;
  // --- for crystal limit spectrum ---
  int num_k = jams::config_required<int>(settings, "num_k");
  std::vector<complex<double>> spectrum_crystal_limit(num_k+1,{0.0,0.0});
  double kmax = jams::config_required<double>(settings, "kmax");
  Vec3 kvector = jams::config_required<Vec3>(settings, "kvector");
  jams::MultiArray<Vec3, 1> k;
  k.resize(num_k+1);
  for (auto n = 0; n < k.size(); ++n) {
      k(n) = kvector * n * (kmax / num_k);
//      cout << "n = " << n << ", kspace_path_(n) = " << k(n) << endl;
  }
  // --- for crystal limit spectrum ---
  for (auto i = 0; i < globals::num_spins; ++i) {
    nbrs.clear();
    ::lattice->atom_neighbours(i, radius_cutoff_, nbrs);

    for (const auto& nbr : nbrs) {
      const auto j = nbr.id;
      if (i == j) {
        continue;
      }
      const auto rij = norm(lattice->displacement(i, j));
      // --- for crystal limit spectrum ---
      const auto rij_vec = lattice->displacement(i, j);
      this->insert_interaction_scalar(i, j, input_unit_conversion_ * exchange_functional(rij));
      counter++;
      for (auto kk = 0; kk < spectrum_crystal_limit.size(); kk++){
          double kr = std::inner_product(k(kk).begin(), k(kk).end(), rij_vec.begin(), 0.0);
          std::complex<double> tmp = { exchange_functional(rij)* (1.0-cos(kTwoPi*kr)),  exchange_functional(rij) * sin(kTwoPi*kr)};
          spectrum_crystal_limit[kk] += tmp;
          if(kr != 0.0){
              counter2++;
          }
      }
      // --- for crystal limit spectrum ---
    }
  }
  for (auto kk = 0; kk < spectrum_crystal_limit.size(); kk++) {
      spectrum_crystal_limit[kk] /= globals::num_spins;
  }

  cout << "  total interactions " << jams::fmt::integer << counter << "\n";
  cout << "  average interactions per spin " << jams::fmt::decimal << counter / double(globals::num_spins) << "\n";
  cout << "  average interactions per spin (kr != 0) " << jams::fmt::decimal << counter2 / double(globals::num_spins)/num_k << "\n";
  // --- for crystal limit spectrum ---
  for (auto m = 0; m < spectrum_crystal_limit.size(); m++) {
//      cout << "  spectrum_crystal_limit (" << m << ") = " << spectrum_crystal_limit[m] << "\n";
//      cout << "  real (" << m << ") = " << spectrum_crystal_limit[m].real() << "\n";
//      cout << "  imag (" << m << ") = " << spectrum_crystal_limit[m].imag() << "\n";
      outfile3 << std::setw(20) << k(m)[0] << "\t";
      outfile3 << std::setw(20) << k(m)[1] << "\t";
      outfile3 << std::setw(20) << k(m)[2] << "\t";
      outfile3 << std::setw(20) << spectrum_crystal_limit[m].real() << "\t";
      outfile3 << std::setw(20) << spectrum_crystal_limit[m].imag() << "\n";
  }
  // --- for crystal limit spectrum ---

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
  return J0 * exp(-pow2(rij - r0)/(2 * pow2(sigma)));
}

double ExchangeFunctionalHamiltonian::functional_kaneyoshi(const double rij, const double J0, const double r0, const double sigma){
  return J0 * pow2(rij - r0) * exp(-pow2(rij - r0) / (2 * pow2(sigma)));
}

double ExchangeFunctionalHamiltonian::functional_step(const double rij, const double J0, const double r_out){
    if (rij < r_out){
        return J0;
    }
    else
        return 0.0;
}

double ExchangeFunctionalHamiltonian::functional_gaussian_multi(const double rij, const double J0, const double r0, const double sigma, const double J0_2, const double r0_2, const double sigma_2, const double J0_3, const double r0_3, const double sigma_3){
    return J0 * exp(-pow2(rij - r0)/(2 * pow2(sigma))) + J0_2 * exp(-pow2(rij - r0_2)/(2 * pow2(sigma_2))) + J0_3 * exp(-pow2(rij - r0_3)/(2 * pow2(sigma_3)));
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
  } else if (functional_name == "gaussian_multi") {
      return bind(functional_gaussian_multi, _1, double(settings["J0"]), double(settings["r0"]), double(settings["sigma"]), double(settings["J0_2"]), double(settings["r0_2"]), double(settings["sigma_2"]), double(settings["J0_3"]), double(settings["r0_3"]), double(settings["sigma_3"]));
  } else if (functional_name == "kaneyoshi") {
      return bind(functional_kaneyoshi, _1, double(settings["J0"]), double(settings["r0"]), double(settings["sigma"]));
  } else if (functional_name == "step") {
      return bind(functional_step, _1, double(settings["J0"]), double(settings["r_out"]));
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


