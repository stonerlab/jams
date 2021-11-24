#ifndef JAMS_HAMILTONIAN_EXCHANGE_FUNCTIONAL_H
#define JAMS_HAMILTONIAN_EXCHANGE_FUNCTIONAL_H

#include <functional>

#include <libconfig.h++>
#include <pcg/pcg_random.hpp>
#include <unordered_map>
#include <random>

#include "jams/helpers/random.h"
#include "jblib/containers/array.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/interactions.h"
#include "jams/containers/sparse_matrix.h"
#include "jams/hamiltonian/sparse_interaction.h"

class ExchangeFunctionalHamiltonian : public SparseInteractionHamiltonian {
public:
    ExchangeFunctionalHamiltonian(const libconfig::Setting &settings, const unsigned int size);
private:
    using ExchangeFunctional = std::function<double(double)>;

    static ExchangeFunctional functional_from_settings(const libconfig::Setting &settings);
    static void output_exchange_functional(std::ostream &os, const ExchangeFunctional& functional, double r_cutoff, double delta_r = 0.001);

    static double functional_rkky(double rij, double J0, double r0, double k_F);
    static double functional_exp(double rij, double J0, double r0, double lengthscale);
    static double functional_gaussian(double rij, double J0, double r0, double lengthscale);
    static double functional_gaussian_multi(double rij, double J0, double r0, double lengthscale, double J0_2, double r0_2, double lengthscale_2, double J0_3, double r0_3, double lengthscale_3);
    static double functional_kaneyoshi(double rij, double J0, double r0, double lengthscale);
    static double functional_step(double rij, double J0, double r_out);
    static double functional_random(const double rij, const double J0, const double r_out, const double width);

    pcg32_k1024 random_generator_ = pcg_extras::seed_seq_from<pcg32>(jams::random_generator());

    double radius_cutoff_;
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_FUNCTIONAL_H