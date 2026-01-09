#include <jams/hamiltonian/cuda_uniaxial_anisotropy.h>
#include <jams/hamiltonian/uniaxial_anisotropy.h>

#include <jams/core/globals.h>

#include "jams/cuda/cuda_array_kernels.h"
#include "jams/cuda/cuda_device_vector_ops.h"

__global__ void cuda_uniaxial_energy_kernel(const int num_spins, const int power,
  const jams::Real * magnitude, const jams::Real * axis, const jams::RealHi * dev_s, jams::Real * dev_e) {
  const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  const jams::Real s[3] = {static_cast<jams::Real>(dev_s[base + 0]), static_cast<jams::Real>(dev_s[base + 1]), static_cast<jams::Real>(dev_s[base + 2])};
  const jams::Real a[3] = {axis[base + 0], axis[base + 1], axis[base + 2]};
  dev_e[idx] = -magnitude[idx] * pow_int(dot(s, a), power);
}

__global__ void cuda_uniaxial_field_kernel(const int num_spins, const int power,
                                           const jams::Real * magnitude, const jams::Real * axis, const jams::RealHi * dev_s, jams::Real * dev_h) {
  const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  const jams::Real3 s{static_cast<jams::Real>(dev_s[base + 0]), static_cast<jams::Real>(dev_s[base + 1]), static_cast<jams::Real>(dev_s[base + 2])};
  const jams::Real3 a{axis[base + 0], axis[base + 1], axis[base + 2]};
  const jams::Real pre = magnitude[idx] * static_cast<jams::Real>(power) * pow_int(dot(s, a), power-1);
  dev_h[base + 0] = pre * a.x;
  dev_h[base + 1] = pre * a.y;
  dev_h[base + 2] = pre * a.z;
}


CudaUniaxialAnisotropyHamiltonian::CudaUniaxialAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialAnisotropyHamiltonian(settings, num_spins)
{
}

jams::Real CudaUniaxialAnisotropyHamiltonian::calculate_total_energy(jams::Real time) {
  calculate_energies(time);
  return cuda_reduce_array(energy_.device_data(), globals::num_spins, cuda_stream_.get());
}

void CudaUniaxialAnisotropyHamiltonian::calculate_energies(jams::Real time) {
  cuda_uniaxial_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
  (globals::num_spins, power_, magnitude_.device_data(), axis_.device_data(), globals::s.device_data(), energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaUniaxialAnisotropyHamiltonian::calculate_fields(jams::Real time) {
  cuda_uniaxial_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
            (globals::num_spins, power_, magnitude_.device_data(), axis_.device_data(), globals::s.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
