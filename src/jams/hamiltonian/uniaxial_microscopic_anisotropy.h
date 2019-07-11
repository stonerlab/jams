// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_UNIAXIAL_MICROSCOPIC_H
#define JAMS_HAMILTONIAN_UNIAXIAL_MICROSCOPIC_H

#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"

class UniaxialMicroscopicHamiltonian : public Hamiltonian {
    friend class CudaUniaxialMicroscopicHamiltonian;
    public:
    UniaxialMicroscopicHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~UniaxialMicroscopicHamiltonian() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();
    private:
        jams::MultiArray<int, 1> mca_order_; // MCA expressed as a Legendre polynomial
        jams::MultiArray<double, 2> mca_value_; // first index in mca order and second is spin index
};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_MICROSCOPIC_H