// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H
#define JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H

#include <jams/core/hamiltonian.h>

#include <vector>
#include <utility>

class DipoleNeighbourListHamiltonian : public Hamiltonian {
public:
    DipoleNeighbourListHamiltonian(const libconfig::Setting &settings, unsigned int size);

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override;

private:
    std::vector<std::vector<std::pair<Vec3R, int>>> neighbour_list_;
    jams::Real r_cutoff_; // cutoff radius for dipole interaction
    jams::Real dipole_prefactor_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H
