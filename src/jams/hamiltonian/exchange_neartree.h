// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H
#define JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H

#include <unordered_map>

#include <libconfig.h++>

#include "jams/containers/sparse_matrix.h"
#include "jams/hamiltonian/sparse_interaction.h"

struct InteractionNT {
    int material[2];
    double inner_radius;
    double outer_radius;
    double value;
};

class ExchangeNeartreeHamiltonian : public SparseInteractionHamiltonian {
public:
    ExchangeNeartreeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    typedef std::vector<std::vector<InteractionNT>> InteractionList;

private:
    InteractionList interaction_list_;
    double energy_cutoff_;
    double distance_tolerance_;
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H