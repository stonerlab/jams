// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_H
#define JAMS_HAMILTONIAN_EXCHANGE_H

#include <libconfig.h++>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "core/output.h"
#include "core/hamiltonian.h"
#include "core/cuda_defs.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class ExchangeHamiltonian : public Hamiltonian {
    public:
        ExchangeHamiltonian(const libconfig::Setting &settings);
        ~ExchangeHamiltonian() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final);
        void   calculate_energies();

        void   calculate_one_spin_fields(const int i, double h[3]);
        void   calculate_fields();

        void   output_energies(OutputFormat format);
        void   output_fields(OutputFormat format);

    private:

        void read_interactions(const std::string &filename,
          std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > &int_interaction_list);

        void read_interactions_with_symmetry(const std::string &filename,
          std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > &int_interaction_list);

        bool insert_interaction(const int m, const int n, const jblib::Matrix<double, 3, 3> &value);

        void output_energies_text();
        // void output_energies_hdf5();

        void output_fields_text();
        // void output_fields_hdf5();

        SparseMatrixFormat_t sparse_matrix_format();
        void set_sparse_matrix_format(std::string &format_name);

        SparseMatrix<double> interaction_matrix_;
        SparseMatrixFormat_t interaction_matrix_format_;
        double energy_cutoff_;


#ifdef CUDA
        devDIA dev_dia_interaction_matrix_;
        devCSR dev_csr_interaction_matrix_;
        cusparseHandle_t   cusparse_handle_;
        cusparseMatDescr_t cusparse_descra_;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H