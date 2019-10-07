// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_H
#define JAMS_HAMILTONIAN_EXCHANGE_H

#include <unordered_map>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"
#include "jams/core/interactions.h"
#include "jams/containers/sparse_matrix.h"
#include "jams/hamiltonian/sparse_interaction.h"

class ExchangeHamiltonian : public SparseInteractionHamiltonian {
    public:
        ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size);

        const InteractionList<Mat3>& neighbour_list() const;

    private:
        InteractionList<Mat3> neighbour_list_;
        double energy_cutoff_;
        double radius_cutoff_;
        double distance_tolerance_;
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H
