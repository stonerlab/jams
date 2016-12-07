// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_CUDA_SPARSE_TENSOR_H
#define JAMS_HAMILTONIAN_DIPOLE_CUDA_SPARSE_TENSOR_H

#ifdef CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "hamiltonian/strategy.h"

typedef struct devFloatCSR {
  int     *row;
  int     *col;
  float   *val;
  int     blocks;
} devFloatCSR;

class DipoleHamiltonianCUDASparseTensor : public HamiltonianStrategy {
    public:
        DipoleHamiltonianCUDASparseTensor(const libconfig::Setting &settings);

        ~DipoleHamiltonianCUDASparseTensor() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) ;
        void   calculate_energies(jblib::Array<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jblib::Array<double, 2>& fields);
        void   calculate_fields(jblib::CudaArray<double, 1>& fields);

    private:
        bool               use_double_precision;

        double r_cutoff_;

        SparseMatrix<float> interaction_matrix_;

        jblib::CudaArray<float, 1> dev_float_spins_;
        jblib::CudaArray<float, 1> dev_float_fields_;

        cudaStream_t       dev_stream_;
        devFloatCSR        dev_csr_interaction_matrix_;
        cusparseHandle_t   cusparse_handle_;
        cusparseMatDescr_t cusparse_descra_;
};

#endif  // JAMS_HAMILTONIAN_DIPOLE_CUDA_SPARSE_TENSOR_H