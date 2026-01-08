// biquadratic_exchange.h                                              -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_BIQUADRATIC_EXCHANGE
#define INCLUDED_JAMS_CUDA_BIQUADRATIC_EXCHANGE
/// @brief:
///
/// @details: This component...
///
/// Usage
/// -----

#include <jams/core/hamiltonian.h>
#include <jams/containers/interaction_list.h>
#include <jams/containers/sparse_matrix_builder.h>


class CudaBiquadraticExchangeHamiltonian : public Hamiltonian {
public:
    CudaBiquadraticExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    jams::Real calculate_total_energy(jams::Real time) override;

    void calculate_fields(jams::Real time) override;

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override;


private:
    double distance_tolerance_; // distance tolerance for calculating interactions
    double energy_cutoff_; // abs cutoff energy for interaction
    double radius_cutoff_; // cutoff radius for interaction

    jams::InteractionList<Mat3, 2> neighbour_list_; // neighbour list

    bool is_finalized_ = false; // is the sparse matrix finalized and built
    jams::SparseMatrix<jams::Real>::Builder sparse_matrix_builder_; // helper to build the sparse matrix and output a chosen type
    jams::SparseMatrix<jams::Real> interaction_matrix_; // the sparse matrix to be used in calculations
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------