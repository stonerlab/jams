//
// Created by Joe Barker on 2018/05/28.
//

#include <jams/hamiltonian/cuda_random_anisotropy.h>
#include <jams/hamiltonian/cuda_random_anisotropy_kernel.cuh>

#include <jams/core/globals.h>
#include <jams/hamiltonian/random_anisotropy.h>
#include <jams/helpers/utils.h>
#include <jams/interface/config.h>

CudaRandomAnisotropyHamiltonian::CudaRandomAnisotropyHamiltonian(const libconfig::Setting &settings,
                                                                 const unsigned int size)
        : RandomAnisotropyHamiltonian(settings, size)
{
  dev_magnitude_ = magnitude_;
  dev_direction_ = flatten_vector(direction_);
}

void CudaRandomAnisotropyHamiltonian::calculate_fields(double time) {
  const unsigned num_blocks = (globals::num_spins+dev_blocksize_-1)/dev_blocksize_;
  random_anisotropy_cuda_field_kernel<<<num_blocks, dev_blocksize_, 0, dev_stream_.get()>>>(
          globals::num_spins,
          globals::s.device_data(),
          dev_direction_.data().get(),
          dev_magnitude_.data().get(),
          field_.device_data()
          );
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaRandomAnisotropyHamiltonian::calculate_energies(double time) {
  const unsigned num_blocks = (globals::num_spins+dev_blocksize_-1)/dev_blocksize_;
  random_anisotropy_cuda_energy_kernel<<<num_blocks, dev_blocksize_, 0, dev_stream_.get()>>>(
          globals::num_spins,
          globals::s.device_data(),
          dev_direction_.data().get(),
          dev_magnitude_.data().get(),
          energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

double CudaRandomAnisotropyHamiltonian::calculate_total_energy(double time) {
  calculate_energies(time);
  return thrust::reduce(energy_.device_data(), energy_.device_data()+energy_.size());
}
