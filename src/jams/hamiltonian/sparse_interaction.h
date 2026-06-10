//
// Created by Joseph Barker on 2019-10-06.
//

#ifndef JAMS_HAMILTONIAN_SPARSE_INTERACTION_H
#define JAMS_HAMILTONIAN_SPARSE_INTERACTION_H

#if HAS_CUDA
#include <jams/cuda/cuda_stream.h>
#endif

#include <jams/core/hamiltonian.h>
#include <jams/containers/block_sparse_interaction_matrix.h>
#include <jams/containers/sparse_matrix_builder.h>

#include "jams/helpers/mixed_precision.h"


class SparseInteractionHamiltonian : public Hamiltonian {
public:
    SparseInteractionHamiltonian(const libconfig::Setting &settings, unsigned int size);

    jams::Real calculate_total_energy(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;

    void calculate_energies(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;

    void calculate_fields(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;

    jams::Vec<jams::Real, 3> calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const jams::Vec<double, 3> &spin_initial, const jams::Vec<double, 3> &spin_final, jams::Real time) override;

    EnergyCurrentInteractionSupport energy_current_interaction_support() const override {
      return EnergyCurrentInteractionSupport::Supported;
    }

    void add_energy_current_interactions(jams::EnergyCurrentInteractionSink& sink) const override;

protected:
    // inserts a scalar interaction into the interaction matrix
    void insert_interaction_scalar(int i, int j, const jams::Real &value);

    // inserts a tensor interaction block into the interaction matrix
    void insert_interaction_tensor(int i, int j, const jams::Mat<jams::Real, 3, 3> &value);

    // finishes constructing the sparse_matrix_builder_ making the builder
    // emit a matrix for use in calculations
    void finalize(jams::SparseMatrixSymmetryCheck symmetry_check);

private:
    bool is_finalized_ = false; // is the sparse matrix finalized and built
    jams::BlockSparseInteractionMatrix<jams::Real>::Builder interaction_matrix_builder_;
    jams::BlockSparseInteractionMatrix<jams::Real> interaction_matrix_;
};

#endif //JAMS_HAMILTONIAN_SPARSE_INTERACTION_H
