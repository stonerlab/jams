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

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_field(int i) override;

    double calculate_energy(int i) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;


private:
    double distance_tolerance_; // distance tolerance for calculating interactions
    double energy_cutoff_; // abs cutoff energy for interaction
    double radius_cutoff_; // cutoff radius for interaction

    jams::InteractionList<Mat3, 2> neighbour_list_; // neighbour list

    bool is_finalized_ = false; // is the sparse matrix finalized and built
    jams::SparseMatrix<double>::Builder sparse_matrix_builder_; // helper to build the sparse matrix and output a chosen type
    jams::SparseMatrix<double> interaction_matrix_; // the sparse matrix to be used in calculations
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------