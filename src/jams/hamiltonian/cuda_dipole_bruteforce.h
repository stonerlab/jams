// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H

#include "jams/helpers/maths.h"
#include "jams/hamiltonian/strategy.h"

#include "jams/cuda/cuda_stream.h"

class CudaDipoleHamiltonianBruteforce : public HamiltonianStrategy {
    public:
        CudaDipoleHamiltonianBruteforce(const libconfig::Setting &settings, const unsigned int size);

        ~CudaDipoleHamiltonianBruteforce();

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const Vec3 &s_i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
        void   calculate_energies(jams::MultiArray<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jams::MultiArray<double, 2>& fields);

    private:
        double r_cutoff_;
        double dipole_prefactor_;

        jams::MultiArray<float, 2> r_float_;
        jams::MultiArray<float, 1> mus_float_;
        jams::MultiArray<double, 2> dipole_fields_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
