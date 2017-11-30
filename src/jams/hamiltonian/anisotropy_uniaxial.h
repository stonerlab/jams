// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_UNIAXIAL_H
#define JAMS_HAMILTONIAN_UNIAXIAL_H

#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class UniaxialHamiltonian : public Hamiltonian {
    public:
        UniaxialHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~UniaxialHamiltonian() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();
    private:
        std::vector<int> mca_order_;   // MCA expressed as a Legendre polynomial
        std::vector< jblib::Array<double, 1> > mca_value_;

#ifdef CUDA
        cudaStream_t dev_stream_ = nullptr;
        unsigned int dev_blocksize_;
        jblib::CudaArray<int, 1> dev_mca_order_;
        jblib::CudaArray<double, 1> dev_mca_value_;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_UNIAXIAL_H