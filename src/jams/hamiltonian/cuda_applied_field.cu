#include <jams/hamiltonian/cuda_applied_field.h>
#include <jams/hamiltonian/cuda_applied_field_kernel.cuh>

CudaAppliedFieldHamiltonian::CudaAppliedFieldHamiltonian(
    const libconfig::Setting &settings, const unsigned int size) : AppliedFieldHamiltonian(
    settings, size) {}

void CudaAppliedFieldHamiltonian::calculate_fields(double time) {
  dim3 block_size;
  block_size.x = 64;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

  const double3 field = {b_field()[0], b_field()[1], b_field()[2]};
  cuda_applied_field_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
        (globals::num_spins, globals::mus.device_data(),
         field, field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaAppliedFieldHamiltonian::calculate_energies(double time) {
  dim3 block_size;
  block_size.x = 64;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

  const double3 field = {b_field()[0], b_field()[1], b_field()[2]};
  cuda_applied_field_energy_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
      (globals::num_spins, globals::s.device_data(), globals::mus.device_data(),
       field, energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}