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
    ExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    const jams::InteractionList<Mat3, 2> &neighbour_list() const;

private:
    jams::InteractionList<Mat3, 2> neighbour_list_; // neighbour information
    double interaction_prefactor_; // prefactor to multiply interactions by to change between Hamiltonian conventions
    double energy_cutoff_; // abs cutoff energy for interaction
    double radius_cutoff_; // cutoff radius for interaction
    double distance_tolerance_; // distance tolerance for calculating interactions
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H
