// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_EXCHANGE_H
#define JAMS_HAMILTONIAN_EXCHANGE_H

#include <unordered_map>

#include <libconfig.h++>

#ifdef CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "core/output.h"
#include "core/hamiltonian.h"
#include "core/cuda_defs.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

struct Interaction {
    int index;
    jblib::Matrix<double, 3, 3> tensor;
};

class ExchangeHamiltonian : public Hamiltonian {
    public:
        ExchangeHamiltonian(const libconfig::Setting &settings);
        ~ExchangeHamiltonian() {};

        typedef std::vector<std::map<int, Mat3>> InteractionList;

        std::string name() const { return "exchange"; }

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final);

        // double calculate_bond_energy(const int i, const int j);
        double calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final);


        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

        void   output_energies(OutputFormat format);
        void   output_fields(OutputFormat format);

        const std::map<int, Mat3>& neighbours(const int i) const {
            return neighbour_list_[i];
        }

    private:

        void read_interactions(const std::string &filename,
          std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > &int_interaction_list);

        void read_interactions_with_symmetry(const std::string &filename,
          std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > &int_interaction_list);

        void insert_interaction(const int i, const int j, const jblib::Matrix<double, 3, 3> &value);

        void output_energies_text();
        // void output_energies_hdf5();

        void output_fields_text();
        // void output_fields_hdf5();

        sparse_matrix_format_t sparse_matrix_format();
        void set_sparse_matrix_format(std::string &format_name);

        InteractionList neighbour_list_;
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
        cudaStream_t dev_stream_;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_EXCHANGE_H