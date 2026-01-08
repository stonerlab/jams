#include <jams/hamiltonian/cuda_applied_field.h>
#include <jams/core/globals.h>
#include <jams/helpers/mixed_precision.h>
#include <jams/cuda/cuda_device_vector_ops.h>
#include <jams/cuda/cuda_array_kernels.h>


__global__ void cuda_applied_field_energy_kernel(const unsigned int num_spins, const double * dev_s, const jams::Real * dev_mus, const jams::Real3 b_field, jams::Real * dev_e) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_spins) return;

    jams::Real3 s = {static_cast<jams::Real>(dev_s[3 * idx + 0]), static_cast<jams::Real>(dev_s[3 * idx + 1]), static_cast<jams::Real>(dev_s[3 * idx + 2])};
    dev_e[idx] = -dev_mus[idx] * dot(s, b_field);
}

__global__ void cuda_applied_field_kernel(const unsigned int num_spins, const jams::Real * dev_mus, const jams::Real3 b_field, jams::Real * dev_h) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_spins) return;

    dev_h[3*idx + 0] = dev_mus[idx] * b_field.x;
    dev_h[3*idx + 1] = dev_mus[idx] * b_field.y;
    dev_h[3*idx + 2] = dev_mus[idx] * b_field.z;
}

CudaAppliedFieldHamiltonian::CudaAppliedFieldHamiltonian(
    const libconfig::Setting &settings, const unsigned int size) : AppliedFieldHamiltonian(
    settings, size) {}

void CudaAppliedFieldHamiltonian::calculate_fields(jams::Real time) {
  dim3 block_size;
  block_size.x = 64;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

  const Vec3R b_field = time_dependent_field_->field(time);
  cuda_applied_field_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
        (globals::num_spins, globals::mus.device_data(),
         {b_field[0], b_field[1], b_field[2]}, field_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaAppliedFieldHamiltonian::calculate_energies(jams::Real time) {
  dim3 block_size;
  block_size.x = 64;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

  const Vec3R b_field = time_dependent_field_->field(time);
  cuda_applied_field_energy_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
      (globals::num_spins, globals::s.device_data(), globals::mus.device_data(),
       {b_field[0], b_field[1], b_field[2]}, energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

jams::Real CudaAppliedFieldHamiltonian::calculate_total_energy(jams::Real time)
{
    calculate_energies(time);
    return cuda_reduce_array(energy_.device_data(), globals::num_spins, cuda_stream_.get());
}
