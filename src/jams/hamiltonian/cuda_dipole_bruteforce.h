// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H

#include "jams/helpers/maths.h"
#include "jams/core/hamiltonian.h"

#include "jams/cuda/cuda_stream.h"

class CudaDipoleBruteforceHamiltonian : public Hamiltonian {
    public:
        CudaDipoleBruteforceHamiltonian(const libconfig::Setting &settings, const unsigned int size);

        ~CudaDipoleBruteforceHamiltonian() = default;

        double calculate_total_energy();
        double calculate_energy(const int i);
        double calculate_one_spin_energy(const int i, const Vec3 &s_i);
        double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
        void   calculate_energies();

        Vec3   calculate_field(const int i);
        void   calculate_fields();

    private:
        double r_cutoff_;
        double dipole_prefactor_;

        jams::MultiArray<float, 2> r_float_;
        jams::MultiArray<float, 1> mus_float_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
