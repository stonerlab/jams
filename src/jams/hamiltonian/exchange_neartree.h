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

/// Hamiltonian for exchange specified by radius
///
/// \f[
///     \mathcal{H} = \frac{1}{2} \sum_{ij} J({r_{ij}) \mathbf{S}_i \cdot \mathbf{S}_j
/// \f]
///
/// where J({r_{ij}) can be defined between different material types.
///
/// Example
/// -------
///
/// hamiltonians = (
/// {
///     module = "exchange-neartree";
///     energy_units = "meV";
///     interactions = (
///         // Shell 1 (r = 1/sqrt(2), 12 nbrs)
///         ("Ni", "Ni", 0.70710678, 17.0),
///         ("Ni", "Fe", 0.70710678, 31.0),
///         ("Fe", "Fe", 0.70710678, 44.0)
/// }
/// );

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