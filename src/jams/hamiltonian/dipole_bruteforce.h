// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_CPU_BRUTEFORCE_H
#define JAMS_HAMILTONIAN_DIPOLE_CPU_BRUTEFORCE_H

#include "jams/helpers/maths.h"
#include "strategy.h"

class DipoleHamiltonianCpuBruteforce : public HamiltonianStrategy {
    public:
        DipoleHamiltonianCpuBruteforce(const libconfig::Setting &settings, const unsigned int size);

        ~DipoleHamiltonianCpuBruteforce();

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const Vec3 &s_i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
        void   calculate_energies(jblib::Array<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jblib::Array<double, 2>& fields);

#if HAS_CUDA
        void   calculate_fields(jblib::CudaArray<double, 1>& fields) {}
#endif

    private:
        std::vector<Vec3>   frac_positions_;
        Mat3 supercell_matrix_;
        double r_cutoff_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H