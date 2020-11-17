#include <jams/hamiltonian/cuda_applied_field.h>
#include <jams/hamiltonian/cuda_applied_field_kernel.cuh>

CudaAppliedFieldHamiltonian::CudaAppliedFieldHamiltonian(
    const libconfig::Setting &settings, const unsigned int size) : AppliedFieldHamiltonian(
    settings, size) {}

void CudaAppliedFieldHamiltonian::calculate_fields() {
  dim3 block_size;
  block_size.x = 32;
  block_size.y = 4;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;
  grid_size.y = (3 + block_size.y - 1) / block_size.y;

  cuda_applied_field_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
        (globals::num_spins, globals::mus.device_data(),
        b_field().data(), field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaAppliedFieldHamiltonian::calculate_energies() {
  dim3 block_size;
  block_size.x = 64;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

  cuda_applied_field_energy_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
      (globals::num_spins, globals::s.device_data(), globals::mus.device_data(),
       b_field().data(), energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}