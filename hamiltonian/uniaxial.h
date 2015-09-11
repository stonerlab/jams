// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_UNIAXIAL_H
#define JAMS_HAMILTONIAN_UNIAXIAL_H

#include <libconfig.h++>

#include "core/output.h"
#include "core/hamiltonian.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class UniaxialHamiltonian : public Hamiltonian {
    public:
        UniaxialHamiltonian(const libconfig::Setting &settings);
        ~UniaxialHamiltonian() {};

        std::string name() const { return "uniaxial"; }

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

        jblib::Array<double, 1> d2z_;
        jblib::Array<double, 1> d4z_;
        jblib::Array<double, 1> d6z_;

#ifdef CUDA
        jblib::CudaArray<double, 1> dev_d2z_;
        jblib::CudaArray<double, 1> dev_d4z_;
        jblib::CudaArray<double, 1> dev_d6z_;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_H