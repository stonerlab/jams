// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_H
#define JAMS_HAMILTONIAN_EXCHANGE_H

#include <unordered_map>

#include <libconfig.h++>

#ifdef CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#include "jams/cuda/wrappers/stream.h"

#endif

#include "jams/core/hamiltonian.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/core/interactions.h"
#include "jams/containers/sparsematrix.h"
#include "jams/cuda/cuda-sparse-helpers.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class ExchangeHamiltonian : public Hamiltonian {
    public:
        ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~ExchangeHamiltonian() = default;

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);
        void   calculate_energies();
        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

    private:
        void insert_interaction(const int i, const int j, const Mat3 &value, const double& energy_cutoff);

        SparseMatrix<double> interaction_matrix_;

#ifdef CUDA
        CudaSparseMatrixCSR<double> dev_csr_interaction_matrix_;
        cusparseHandle_t   cusparse_handle_;
        CudaStream dev_stream_;
#endif  // CUDA
};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H