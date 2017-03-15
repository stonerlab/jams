// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_ZEEMAN_H
#define JAMS_HAMILTONIAN_ZEEMAN_H

#include <libconfig.h++>

#include "jams/core/output.h"
#include "jams/core/hamiltonian.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class ZeemanHamiltonian : public Hamiltonian {
    public:
        ZeemanHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~ZeemanHamiltonian() {};

        std::string name() const { return "zeeman"; }

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final);
        
        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();
    
    private:
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