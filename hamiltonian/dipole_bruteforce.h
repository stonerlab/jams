// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H

#include "hamiltonian/strategy.h"


class DipoleHamiltonianBruteforce : public HamiltonianStrategy {
    public:
        DipoleHamiltonianBruteforce(const libconfig::Setting &settings);

        ~DipoleHamiltonianBruteforce() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) ;
        void   calculate_energies(jblib::Array<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jblib::Array<double, 2>& fields);
    private:
        double r_cutoff_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_H