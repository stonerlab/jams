// biquadratic_exchange.h                                              -*-C++-*-
#ifndef INCLUDED_JAMS_BIQUADRATIC_EXCHANGE
#define INCLUDED_JAMS_BIQUADRATIC_EXCHANGE

#include <jams/core/hamiltonian.h>
#include <jams/containers/interaction_list.h>
#include <jams/containers/sparse_matrix_builder.h>

///
/// Hamiltonian for biquadratic exchange
///
/// \f[
///     \mathcal{H} = -\frac{1}{2}\sum_{ij} B_{ij} (\vec{S}_{i} \cdot \vec{S}_j)^2
/// \f]
///
/// @details Implements the biquadratic exchange Hamiltonian above. The factor
/// 1/2 accounts for double counting in the sum. Here B_ij is a scalar, not
/// a tensor.
///
/// The effective field from the Hamiltonian is
///
/// \f[
///     \vec{H}_i = 2 B_{ij} \vec{S}_j (\vec{S}_{i} \cdot \vec{S}_j)
/// \f]
///
/// @warning If a full tensor is specified for the interactions
/// in the config, only the first element will be used as the scalar value.
///

class BiquadraticExchangeHamiltonian : public Hamiltonian {
public:
    BiquadraticExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;


protected:
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