//
// Created by Joe Barker on 2018/05/28.
//

#include <jams/hamiltonian/cuda_random_anisotropy.h>
#include <jams/core/globals.h>
#include <jams/hamiltonian/random_anisotropy.h>
#include <jams/helpers/utils.h>
#include <jams/interface/config.h>
#include "jams/cuda/cuda_device_vector_ops.h"

__global__
void random_anisotropy_cuda_field_kernel(
        const int num_spins,
        const jams::RealHi * spins,
        const jams::Real * directions,
        const jams::Real * magnitudes,
        jams::Real * fields)
{
  const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  const jams::Real s[3] = {static_cast<jams::Real>(spins[base + 0]), static_cast<jams::Real>(spins[base + 1]), static_cast<jams::Real>(spins[base + 2])};
  const jams::Real x[3] = {directions[base + 0], directions[base + 1], directions[base + 2]};
  const jams::Real d = magnitudes[idx];

#pragma unroll
  for (auto n = 0; n < 3; ++n) {
    fields[3 * idx + n] = d * dot(x, s) * x[n];
  }
}

__global__
void random_anisotropy_cuda_energy_kernel(
        const int num_spins,
        const jams::RealHi * spins,
        const jams::Real * directions,
        const jams::Real * magnitudes,
        jams::Real * energies)
{
  const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  const jams::Real s[3] = {static_cast<jams::Real>(spins[base + 0]), static_cast<jams::Real>(spins[base + 1]), static_cast<jams::Real>(spins[base + 2])};
  const jams::Real x[3] = {directions[base], directions[base + 1], directions[base + 2]};
  const jams::Real d = magnitudes[idx];

  energies[idx] = -d * dot(x, s) * dot(x, s);
}


CudaRandomAnisotropyHamiltonian::CudaRandomAnisotropyHamiltonian(const libconfig::Setting &settings,
                                                                 const unsigned int size)
        : RandomAnisotropyHamiltonian(settings, size)
{
  dev_magnitude_ = magnitude_;
  dev_direction_ = flatten_vector(direction_);
}

void CudaRandomAnisotropyHamiltonian::calculate_fields(jams::Real time) {
  const unsigned num_blocks = (globals::num_spins+dev_blocksize_-1)/dev_blocksize_;
  random_anisotropy_cuda_field_kernel<<<num_blocks, dev_blocksize_, 0, cuda_stream_.get()>>>(
          globals::num_spins,
          globals::s.device_data(),
          dev_direction_.data().get(),
          dev_magnitude_.data().get(),
          field_.device_data()
          );
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaRandomAnisotropyHamiltonian::calculate_energies(jams::Real time) {
  const unsigned num_blocks = (globals::num_spins+dev_blocksize_-1)/dev_blocksize_;
  random_anisotropy_cuda_energy_kernel<<<num_blocks, dev_blocksize_, 0, cuda_stream_.get()>>>(
          globals::num_spins,
          globals::s.device_data(),
          dev_direction_.data().get(),
          dev_magnitude_.data().get(),
          energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

jams::Real CudaRandomAnisotropyHamiltonian::calculate_total_energy(jams::Real time) {
  calculate_energies(time);
  return thrust::reduce(energy_.device_data(), energy_.device_data()+energy_.size());
}
