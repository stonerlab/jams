//
// Created by Joseph Barker on 2019-10-06.
//

#ifndef JAMS_HAMILTONIAN_SPARSE_INTERACTION_H
#define JAMS_HAMILTONIAN_SPARSE_INTERACTION_H

#if HAS_CUDA
#include <cusparse.h>
#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_common.h"
#endif

#include <jams/containers/sparse_matrix_builder.h>
#include "jams/core/hamiltonian.h"

class SparseInteractionHamiltonian : public Hamiltonian {
public:
    SparseInteractionHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~SparseInteractionHamiltonian() override;

    double calculate_total_energy() override;
    double calculate_one_spin_energy(const int i) override;
    double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;
    void   calculate_energies() override;
    void   calculate_one_spin_field(const int i, double h[3]) override;
    void   calculate_fields() override;

protected:
    void insert_interaction_tensor(const int i, const int j, const Mat3 &value);
    void finalize();

private:
    bool is_finalized_ = false;
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_;
    jams::SparseMatrix<double> interaction_matrix_;

    #if HAS_CUDA
    cusparseHandle_t cusparse_handle_ = nullptr;
    CudaStream cusparse_stream_;
    #endif
};

#endif //JAMS_HAMILTONIAN_SPARSE_INTERACTION_H
