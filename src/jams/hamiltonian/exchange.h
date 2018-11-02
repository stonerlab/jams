// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_H
#define JAMS_HAMILTONIAN_EXCHANGE_H

#include <unordered_map>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"
#include "jams/core/interactions.h"
#include "jams/containers/sparsematrix.h"

#include "jblib/containers/array.h"

class ExchangeHamiltonian : public Hamiltonian {
    friend class CudaExchangeHamiltonian;
    public:
        ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~ExchangeHamiltonian() = default;

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);
        void   calculate_energies();
        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

    private:
        void insert_interaction(const int i, const int j, const Mat3 &value);

        InteractionList<Mat3> neighbour_list_;
        SparseMatrix<double> interaction_matrix_;
        double energy_cutoff_;
        double radius_cutoff_;
        double distance_tolerance_;
        InteractionFileFormat exchange_file_format_;
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H