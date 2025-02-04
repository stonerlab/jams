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
#include <jams/containers/sparse_matrix_builder.h>

#include <jams/hamiltonian/neighbour_list_interaction.h>

class CudaBiquadraticExchangeHamiltonian : public NeighbourListInteractionHamiltonian {
public:
    CudaBiquadraticExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

private:
    jams::InteractionList<Mat3, 2> neighbour_list_; // neighbour information

};

#endif
// ----------------------------- END-OF-FILE ----------------------------------