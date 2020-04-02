// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_CPU_BRUTEFORCE_H
#define JAMS_HAMILTONIAN_DIPOLE_CPU_BRUTEFORCE_H

#include "jams/helpers/maths.h"
#include "jams/core/hamiltonian.h"
#include "jams/containers/neartree.h"

class DipoleHamiltonianCpuBruteforce : public Hamiltonian {
    public:
        DipoleHamiltonianCpuBruteforce(const libconfig::Setting &settings, const unsigned int size);

        ~DipoleHamiltonianCpuBruteforce();

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const Vec3 &s_i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

    private:
        using NeartreeFunctorType = std::function<float(const std::pair<Vec3, int>& a, const std::pair<Vec3, int>& b)>;
        NearTree<std::pair<Vec3, int>, NeartreeFunctorType>* near_tree_;
        std::vector<Vec3>   frac_positions_;
        Mat3 supercell_matrix_;
        double r_cutoff_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
