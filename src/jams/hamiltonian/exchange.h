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
    jams::RealHi interaction_prefactor_; // prefactor to multiply interactions by to change between Hamiltonian conventions
    jams::RealHi energy_cutoff_; // abs cutoff energy for interaction
    jams::RealHi radius_cutoff_; // cutoff radius for interaction
    jams::RealHi distance_tolerance_; // distance tolerance for calculating interactions
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H
