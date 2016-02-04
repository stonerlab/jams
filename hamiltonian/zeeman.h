// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_ZEEMAN_H
#define JAMS_HAMILTONIAN_ZEEMAN_H

#include <libconfig.h++>

#include "core/output.h"
#include "core/hamiltonian.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class ZeemanHamiltonian : public Hamiltonian {
    public:
        ZeemanHamiltonian(const libconfig::Setting &settings);
        ~ZeemanHamiltonian() {};

        std::string name() const { return "zeeman"; }

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final);

        double calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final);

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

        jblib::Array<double, 2> dc_local_field_;
        jblib::Array<double, 2> ac_local_field_;
        jblib::Array<double, 1> ac_local_frequency_;

        bool has_ac_local_field_;

#ifdef CUDA
        cudaStream_t dev_stream_;
        jblib::CudaArray<double, 1> dev_dc_local_field_;
        jblib::CudaArray<double, 1> dev_ac_local_field_;
        jblib::CudaArray<double, 1> dev_ac_local_frequency_;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_ZEEMAN_H