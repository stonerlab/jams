#include <iosfwd>

#include <cuda_runtime_api.h>

#include "jams/core/solver.h"
#include "jams/cuda/cuda_sparse_interaction_matrix.h"
#include "jams/hamiltonian/cuda_exchange.h"

CudaExchangeHamiltonian::CudaExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: ExchangeHamiltonian(settings, size)
{
    dev_interaction_matrix_.create_matrix(interaction_matrix_);
    dev_interaction_matrix_.set_cuda_stream(dev_stream_.get());
}

double CudaExchangeHamiltonian::calculate_total_energy() {
  double total_energy = 0.0;
  calculate_fields();
  for (auto i = 0; i < globals::num_spins; ++i) {
    total_energy += -(  globals::s(i,0)*field_(i,0)
                        + globals::s(i,1)*field_(i,1)
                        + globals::s(i,2)*field_(i,2) );
  }
  return 0.5*total_energy;
}

void CudaExchangeHamiltonian::calculate_fields() {
  dev_interaction_matrix_.calculate_fields(globals::s.device_data(), dev_ptr_field());
}