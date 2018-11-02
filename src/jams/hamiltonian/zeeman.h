// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_ZEEMAN_H
#define JAMS_HAMILTONIAN_ZEEMAN_H

#include <libconfig.h++>
#include "jams/core/hamiltonian.h"
#include "jblib/containers/array.h"

class ZeemanHamiltonian : public Hamiltonian {
    friend class CudaZeemanHamiltonian;
    public:
        ZeemanHamiltonian(const libconfig::Setting &settings, const unsigned int size);

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);
        
        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();
    
    private:
        jblib::Array<double, 2> dc_local_field_;
        jblib::Array<double, 2> ac_local_field_;
        jblib::Array<double, 1> ac_local_frequency_;

        bool has_ac_local_field_;
};

#endif  // JAMS_HAMILTONIAN_ZEEMAN_H