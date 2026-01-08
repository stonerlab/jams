// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_UNIAXIAL_H
#define JAMS_HAMILTONIAN_UNIAXIAL_H

#include <jams/core/hamiltonian.h>
#include <jams/containers/multiarray.h>

#include <vector>

class UniaxialAnisotropyHamiltonian : public Hamiltonian {
    friend class CudaUniaxialAnisotropyHamiltonian;

public:
    UniaxialAnisotropyHamiltonian(const libconfig::Setting &settings, unsigned int size);

    void calculate_energies(jams::Real time) override;

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override;

private:
    int power_; // anisotropy power exponent
    jams::MultiArray<jams::Real, 2> axis_; // local uniaxial anisotropy axis
    jams::MultiArray<jams::Real, 1> magnitude_; // magnitude of local anisotropy
};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_H