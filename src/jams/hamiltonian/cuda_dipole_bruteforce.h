// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H

#include "jblib/containers/cuda_array.h"
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

        void   calculate_fields(jblib::CudaArray<double, 1>& fields);

    private:
        double r_cutoff_;
        double dipole_prefactor_;

        jblib::CudaArray<float, 1> dev_r_;
        jblib::CudaArray<float, 1> dev_mus_;
        jblib::CudaArray<double, 1> dev_dipole_fields;
        jblib::Array<double, 2> host_dipole_fields;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
