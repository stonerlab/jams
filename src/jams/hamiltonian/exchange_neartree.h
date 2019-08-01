// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H
#define JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H

#include <unordered_map>

#include <libconfig.h++>

#include "jams/containers/sparsematrix.h"
#include "jams/core/hamiltonian.h"

struct InteractionNT {
    int material[2];
    double inner_radius;
    double outer_radius;
    double value;
};

class ExchangeNeartreeHamiltonian : public Hamiltonian {
    friend class CudaExchangeNeartreeHamiltonian;

    public:
        ExchangeNeartreeHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~ExchangeNeartreeHamiltonian() = default;

        typedef std::vector<std::vector<InteractionNT>> InteractionList;

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();
    private:

        void insert_interaction(const int i, const int j, const Mat3 &value);

        InteractionList interaction_list_;
        SparseMatrix<double> interaction_matrix_;
        double energy_cutoff_;
        double distance_tolerance_;
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H