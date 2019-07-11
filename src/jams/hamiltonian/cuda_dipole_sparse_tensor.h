// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_CUDA_SPARSE_TENSOR_H
#define JAMS_HAMILTONIAN_DIPOLE_CUDA_SPARSE_TENSOR_H

#if HAS_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include "strategy.h"
#include "jams/containers/sparsematrix.h"

typedef struct devFloatCSR {
  int     *row;
  int     *col;
  float   *val;
  int     blocks;
} devFloatCSR;

#if HAS_CUDA
class CudaDipoleHamiltonianSparseTensor : public HamiltonianStrategy {
    public:
        CudaDipoleHamiltonianSparseTensor(const libconfig::Setting &settings, const unsigned int size);

        ~CudaDipoleHamiltonianSparseTensor();

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const Vec3 &s_i);
        double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) ;
        void   calculate_energies(jams::MultiArray<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jams::MultiArray<double, 2>& fields);

    private:
        bool               use_double_precision;

        double r_cutoff_;

        SparseMatrix<float> interaction_matrix_;

        jams::MultiArray<float, 2> float_spins_;
        jams::MultiArray<float, 2> float_fields_;

        cudaStream_t       dev_stream_ = nullptr;
        devFloatCSR        dev_csr_interaction_matrix_;
        cusparseHandle_t   cusparse_handle_;
        cusparseMatDescr_t cusparse_descra_;
};
#endif

#endif  // JAMS_HAMILTONIAN_DIPOLE_CUDA_SPARSE_TENSOR_H