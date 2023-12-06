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

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

private:
    jams::MultiArray<int, 1> mca_order_; // MCA expressed as a Legendre polynomial
    jams::MultiArray<double, 2> mca_value_; // first index in mca order and second is spin index
};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_MICROSCOPIC_H