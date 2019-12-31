// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_UNIAXIAL_H
#define JAMS_HAMILTONIAN_UNIAXIAL_H

#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"

class UniaxialHamiltonian : public Hamiltonian {
    friend class CudaUniaxialHamiltonian;
    public:
        UniaxialHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~UniaxialHamiltonian() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();
    private:
        int power_;
        jams::MultiArray<double, 2>   axis_;
        jams::MultiArray<double, 1>   magnitude_;
};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_H