// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_UNIAXIAL_MICROSCOPIC_H
#define JAMS_HAMILTONIAN_UNIAXIAL_MICROSCOPIC_H

#include <jams/core/hamiltonian.h>
#include <jams/containers/multiarray.h>

#include <vector>

class UniaxialMicroscopicAnisotropyHamiltonian : public Hamiltonian {
    friend class CudaUniaxialMicroscopicAnisotropyHamiltonian;

public:
    UniaxialMicroscopicAnisotropyHamiltonian(const libconfig::Setting &settings, unsigned int size);

    void calculate_energies(jams::Real time) override;

    void calculate_fields(jams::Real time) override;

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override;

private:
    jams::MultiArray<int, 1> mca_order_; // MCA expressed as a Legendre polynomial
    jams::MultiArray<jams::Real, 2> mca_value_; // first index in mca order and second is spin index
};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_MICROSCOPIC_H