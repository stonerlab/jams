// biquadratic_exchange.h                                              -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_BIQUADRATIC_EXCHANGE
#define INCLUDED_JAMS_CUDA_BIQUADRATIC_EXCHANGE
/// @brief:
///
/// @details: This component...
///
/// Usage
/// -----

#include <jams/hamiltonian/biquadratic_exchange.h>
#include <jams/containers/interaction_list.h>
#include <jams/containers/sparse_matrix_builder.h>


class CudaBiquadraticExchangeHamiltonian : public BiquadraticExchangeHamiltonian {
public:
    CudaBiquadraticExchangeHamiltonian(const libconfig::Setting &settings, unsigned int size);

    void calculate_fields(double time) override;
};

#endif
// ----------------------------- END-OF-FILE ----------------------------------