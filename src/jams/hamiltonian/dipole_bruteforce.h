// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_CPU_BRUTEFORCE_H
#define JAMS_HAMILTONIAN_DIPOLE_CPU_BRUTEFORCE_H

#include "jams/helpers/maths.h"
#include "jams/core/hamiltonian.h"
#include "jams/containers/neartree.h"

class DipoleHamiltonianCpuBruteforce : public Hamiltonian {
public:
    DipoleHamiltonianCpuBruteforce(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_one_spin_field(int i) override;

    double calculate_one_spin_energy(int i) override;

    double calculate_one_spin_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

private:
    using NeartreeFunctorType = std::function<double(const std::pair<Vec3, int> &a, const std::pair<Vec3, int> &b)>;

    double r_cutoff_; // cutoff radius for dipole interaction
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
