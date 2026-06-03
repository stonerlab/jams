// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H
#define JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H

#include <jams/core/hamiltonian.h>

#include <vector>
#include <utility>

class DipoleNeighbourListHamiltonian : public Hamiltonian {
public:
    DipoleNeighbourListHamiltonian(const libconfig::Setting &settings, unsigned int size);

    jams::Vec<jams::Real, 3> calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const jams::Vec<double, 3> &spin_initial, const jams::Vec<double, 3> &spin_final, jams::Real time) override;

    void add_energy_current_interactions(jams::SparseMatrix<double>::Builder& rx_builder,
                                         jams::SparseMatrix<double>::Builder& ry_builder,
                                         jams::SparseMatrix<double>::Builder& rz_builder) const override;

private:
    std::vector<std::vector<std::pair<jams::Vec<jams::Real, 3>, int>>> neighbour_list_;
    jams::Real r_cutoff_; // cutoff radius for dipole interaction
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H
