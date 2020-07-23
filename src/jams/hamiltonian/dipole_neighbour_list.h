// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H
#define JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H

#include "jams/helpers/maths.h"
#include "jams/core/hamiltonian.h"
#include "jams/containers/neartree.h"

class DipoleNeighbourListHamiltonian : public Hamiltonian {
public:
    DipoleNeighbourListHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_field(int i) override;

    double calculate_energy(int i) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

private:
    std::vector<std::vector<std::pair<Vec3, int>>> neighbour_list_;
    double r_cutoff_; // cutoff radius for dipole interaction
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_NEIGHBOUR_LIST_H
