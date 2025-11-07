// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H
#define JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H

#include <jams/hamiltonian/sparse_interaction.h>
#include <jams/containers/interaction_list.h>

struct InteractionNT {
    std::pair<int, int> material;
    double rij;
    double Jij;
};

class ExchangeNeartreeHamiltonian : public SparseInteractionHamiltonian {
public:
    ExchangeNeartreeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    typedef std::vector<std::vector<InteractionNT>> InteractionList;

private:
    InteractionList interaction_list_;
    double energy_cutoff_;
    double shell_width_;
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H