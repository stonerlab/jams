#ifndef JAMS_HAMILTONIAN_EXCHANGE_FUNCTIONAL_H
#define JAMS_HAMILTONIAN_EXCHANGE_FUNCTIONAL_H

#include <jams/hamiltonian/sparse_interaction.h>

#include <functional>

// Example config input
//
// hamiltonians = (
//   {
//     module = "exchange-functional";
//     energy_units = "meV";
//     distance_units = "angstroms";
//     interactions = (
//       # type_i, type_j, functional, r_cutoff, parameters...
//       ("Co", "Co", "exponential", 3.0, 4.0, 2.0, 0.5),
//       ("Co", "Gd", "exponential", 3.0,-2.0, 2.0, 0.5),
//       ("Gd", "Gd", "exponential", 3.0, 0.1, 2.0, 0.5)
//     );
//   }
// );


class ExchangeFunctionalHamiltonian : public SparseInteractionHamiltonian {
public:
    ExchangeFunctionalHamiltonian(const libconfig::Setting &settings, unsigned int size);

private:
    using ExchangeFunctionalType = std::function<double(double)>;

    ExchangeFunctionalType functional_from_params(const std::string& name, const std::vector<double>& params);

    void output_exchange_functional(std::ostream &os, const ExchangeFunctionalType &functional, double r_cutoff,
                                           double delta_r=0.001);

    static double functional_step(double rij, double J0, double r_cut);

    static double functional_rkky(double rij, double J0, double r0, double k_F);

    static double functional_exp(double rij, double J0, double r0, double lengthscale);

    static double functional_gaussian(double rij, double J0, double r0, double lengthscale);

    static double functional_gaussian_multi(double rij, double J0, double r0, double sigma0, double J1, double r1, double sigma1, double J2, double r2, double sigma2);

    static double functional_kaneyoshi(double rij, double J0, double r0, double lengthscale);
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_FUNCTIONAL_H