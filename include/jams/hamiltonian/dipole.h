// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_H
#define JAMS_HAMILTONIAN_DIPOLE_H

#include <iosfwd>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/output.h"
#include "jams/core/hamiltonian.h"

#include "jams/hamiltonian/strategy.h"

#include "jblib/containers/array.h"

#include "jblib/containers/cuda_array.h"

class DipoleHamiltonian : public Hamiltonian {
    public:
        DipoleHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~DipoleHamiltonian() {};

        std::string name() const { return "dipole"; }

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

    private:
        HamiltonianStrategy * select_strategy(const libconfig::Setting &settings, const unsigned int size);
        HamiltonianStrategy *dipole_strategy_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_H