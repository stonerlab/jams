// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_H
#define JAMS_HAMILTONIAN_DIPOLE_H

#include <libconfig.h++>

#include "core/output.h"
#include "core/hamiltonian.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class DipoleHamiltonian : public Hamiltonian {
    public:
        DipoleHamiltonian(const libconfig::Setting &settings);
        ~DipoleHamiltonian() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final);
        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

        void   output_energies(OutputFormat format);
        void   output_fields(OutputFormat format);

    private:

        void output_energies_text();
        // void output_energies_hdf5();

        void output_fields_text();
        // void output_fields_hdf5();

        OutputFormat            outformat_;
        jblib::Array<double, 1> energy_;
        jblib::Array<double, 2> field_;

#ifdef CUDA
        jblib::CudaArray<double, 1> dev_energy_;
        jblib::CudaArray<double, 1> dev_field_;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_DIPOLE_H