// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H
#define JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H

#include <unordered_map>

#include <libconfig.h++>

#ifdef CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "jams/core/output.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/cuda_defs.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

struct InteractionNT {
    int material[2];
    double radius;
    double value;
};

class ExchangeNeartreeHamiltonian : public Hamiltonian {
    public:
        ExchangeNeartreeHamiltonian(const libconfig::Setting &settings, const unsigned int size);
        ~ExchangeNeartreeHamiltonian();

        typedef std::vector<std::vector<InteractionNT>> InteractionList;

        std::string name() const { return "exchange"; }

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final);

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();
    private:

        void insert_interaction(const int i, const int j, const jblib::Matrix<double, 3, 3> &value);

        sparse_matrix_format_t sparse_matrix_format();
        void set_sparse_matrix_format(std::string &format_name);

        InteractionList interaction_list_;
        SparseMatrix<double> interaction_matrix_;
        sparse_matrix_format_t interaction_matrix_format_;
        double energy_cutoff_;
        double distance_tolerance_;
        bool is_debug_enabled_;


#ifdef CUDA
        devDIA dev_dia_interaction_matrix_;
        devCSR dev_csr_interaction_matrix_;
        cusparseHandle_t   cusparse_handle_;
        cusparseMatDescr_t cusparse_descra_;
        cudaStream_t dev_stream_ = nullptr;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_DISTANCE_H