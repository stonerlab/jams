// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_UNIAXIAL_H
#define JAMS_HAMILTONIAN_UNIAXIAL_H

#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"

class UniaxialHamiltonian : public Hamiltonian {
    friend class CudaUniaxialHamiltonian;

public:
    UniaxialHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_field(int i) override;

    double calculate_energy(int i) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

    double calculate_internal_energy_difference(int i) override;

    double calculate_total_internal_energy_difference() override;

    double calculate_entropy(int i) override;

    double calculate_total_entropy() override;

private:
    int power_; // anisotropy power exponent
    jams::MultiArray<double, 2> axis_; // local uniaxial anisotropy axis
    jams::MultiArray<double, 1> magnitude_; // magnitude of local anisotropy
};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_H