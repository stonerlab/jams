// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_H
#define JAMS_HAMILTONIAN_EXCHANGE_H

#include <jams/hamiltonian/sparse_interaction.h>
#include <jams/containers/interaction_list.h>

class ExchangeHamiltonian : public SparseInteractionHamiltonian {
public:
    ExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    const jams::InteractionList<Mat3, 2> &neighbour_list() const;

private:
    jams::InteractionList<Mat3, 2> neighbour_list_; // neighbour information
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H
