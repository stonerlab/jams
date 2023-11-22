// biquadratic_exchange.cu                                             -*-C++-*-

#include <jams/hamiltonian/cuda_biquadratic_exchange.h>
#include <jams/hamiltonian/cuda_biquadratic_exchange_kernel.cuh>

#include <jams/hamiltonian/biquadratic_exchange.h>

#include <jams/core/globals.h>
#include <jams/core/interactions.h>

CudaBiquadraticExchangeHamiltonian::CudaBiquadraticExchangeHamiltonian(
    const libconfig::Setting &settings, unsigned int size)
: BiquadraticExchangeHamiltonian(settings, size)
{}


void CudaBiquadraticExchangeHamiltonian::calculate_fields(double time) {
  assert(is_finalized_);

  const dim3 block_size = {128, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_biquadratic_exchange_field_kernel<<<grid_size, block_size>>>
      (globals::num_spins, globals::s.device_data(), interaction_matrix_.row_device_data(),
       interaction_matrix_.col_device_data(), interaction_matrix_.val_device_data(),
       field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

}



