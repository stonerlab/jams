// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DEMAG_H
#define JAMS_HAMILTONIAN_DEMAG_H

#include <libconfig.h++>
#include <cublas_v2.h>

#include "core/output.h"
#include "core/hamiltonian.h"

#include "jblib/containers/vec.h"
#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class DemagHamiltonian : public Hamiltonian {
    public:
        DemagHamiltonian(const libconfig::Setting &settings);
        ~DemagHamiltonian() {};

        std::string name() const { return "demag"; }

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

        jblib::Vec3<double> factors_;


#ifdef CUDA
        cudaStream_t dev_stream_;

        unsigned int dev_blocksize_;

        jblib::CudaArray<double, 1> dev_mus_;
        jblib::CudaArray<double, 1> dev_moments_;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_DEMAG_H