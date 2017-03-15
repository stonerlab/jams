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
        DipoleHamiltonian(const libconfig::Setting &settings);
        ~DipoleHamiltonian() {};

        std::string name() const { return "dipole"; }

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final);

        double calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final) {return 0.0;};

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

        void   output_energies(OutputFormat format);
        void   output_fields(OutputFormat format);

    private:

        HamiltonianStrategy * select_strategy(const libconfig::Setting &settings);
        void output_energies_text();
        // void output_energies_hdf5();

        void output_fields_text();
        // void output_fields_hdf5();

        HamiltonianStrategy *dipole_strategy_;

        OutputFormat            outformat_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_H