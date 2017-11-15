// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_H
#define JAMS_HAMILTONIAN_EXCHANGE_H

#include <unordered_map>

#include <libconfig.h++>

#ifdef CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "jams/core/output.h"
#include "jams/core/hamiltonian.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/core/interactions.h"
#include "jams/containers/sparsematrix.h"
#include "../cuda/cuda-sparse-helpers.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class ExchangeHamiltonian : public Hamiltonian {
    public:
        ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~ExchangeHamiltonian();

        std::string name() const { return "exchange"; }

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final);

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

        InteractionList<Mat3>::const_reference neighbours(const int i) const {
            return neighbour_list_.interactions(i);
        }

    private:
        void insert_interaction(const int i, const int j, const Mat3 &value);

        sparse_matrix_format_t sparse_matrix_format();
        void set_sparse_matrix_format(std::string &format_name);

        InteractionList<Mat3> neighbour_list_;
        SparseMatrix<double> interaction_matrix_;
        sparse_matrix_format_t interaction_matrix_format_;
        double energy_cutoff_;
        double radius_cutoff_;
        double distance_tolerance_;
        bool is_debug_enabled_;
        InteractionFileFormat exchange_file_format_;


#ifdef CUDA
        devDIA dev_dia_interaction_matrix_;
        CudaSparseMatrixCSR<double> dev_csr_interaction_matrix_;
        cusparseHandle_t   cusparse_handle_;
        cudaStream_t dev_stream_ = nullptr;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H