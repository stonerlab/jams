// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_TENSOR_H
#define JAMS_HAMILTONIAN_DIPOLE_TENSOR_H

#include <libconfig.h++>

#include "jblib/containers/array.h"

#include "jams/hamiltonian/strategy.h"


class DipoleHamiltonianTensor : public HamiltonianStrategy {
    public:
        DipoleHamiltonianTensor(const libconfig::Setting &settings, const unsigned int size);

        ~DipoleHamiltonianTensor() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const Vec3 &s_i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
        void   calculate_energies(jblib::Array<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jblib::Array<double, 2>& fields);
    private:
        double r_cutoff_;
        jblib::Array<double, 2> dipole_tensor_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_TENSOR_H