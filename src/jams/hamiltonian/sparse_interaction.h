//
// Created by Joseph Barker on 2019-10-06.
//

#ifndef JAMS_HAMILTONIAN_SPARSE_INTERACTION_H
#define JAMS_HAMILTONIAN_SPARSE_INTERACTION_H

#if HAS_CUDA
#include <jams/cuda/cuda_stream.h>
#endif

#include <jams/core/hamiltonian.h>
#include <jams/containers/sparse_matrix.h>
#include <jams/containers/sparse_matrix_builder.h>

class SparseInteractionHamiltonian : public Hamiltonian {
public:
    SparseInteractionHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

protected:
    // inserts a scalar interaction into the interaction matrix
    void insert_interaction_scalar(int i, int j, const double &value);

    // inserts a tensor interaction block into the interaction matrix
    void insert_interaction_tensor(int i, int j, const Mat3 &value);

    // finishes constructing the sparse_matrix_builder_ making the builder
    // emit a matrix for use in calculations
    void finalize(jams::SparseMatrixSymmetryCheck symmetry_check);

private:
    bool is_finalized_ = false; // is the sparse matrix finalized and built
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_; // helper to build the sparse matrix and output a chosen type
    jams::SparseMatrix<double> interaction_matrix_; // the sparse matrix to be used in calculations

    #if HAS_CUDA
    CudaStream cusparse_stream_; // cuda stream to run in
    #endif
};

#endif //JAMS_HAMILTONIAN_SPARSE_INTERACTION_H
