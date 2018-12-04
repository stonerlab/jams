//
// Created by Joe Barker on 2018/05/28.
//
#include <libconfig.h++>

#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/hamiltonian/cuda_random_anisotropy.h"
#include "jams/hamiltonian/cuda_random_anisotropy_kernel.cuh"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/helpers/utils.h"

CudaRandomAnisotropyHamiltonian::CudaRandomAnisotropyHamiltonian(const libconfig::Setting &settings,
                                                                 const unsigned int size)
        : RandomAnisotropyHamiltonian(settings, size)
{
  dev_magnitude_ = magnitude_;
  dev_direction_ = flatten_vector(direction_);
  dev_energy_    = jblib::CudaArray<double, 1>(energy_);
  dev_field_     = jblib::CudaArray<double, 1>(field_);
}

void CudaRandomAnisotropyHamiltonian::calculate_fields() {
  const unsigned num_blocks = (globals::num_spins+dev_blocksize_-1)/dev_blocksize_;
  random_anisotropy_cuda_field_kernel<<<num_blocks, dev_blocksize_, 0, dev_stream_.get()>>>(
          globals::num_spins,
          solver->dev_ptr_spin(),
          dev_direction_.data().get(),
          dev_magnitude_.data().get(),
          dev_field_.data()
          );
}

void CudaRandomAnisotropyHamiltonian::calculate_energies() {
  const unsigned num_blocks = (globals::num_spins+dev_blocksize_-1)/dev_blocksize_;
  random_anisotropy_cuda_energy_kernel<<<num_blocks, dev_blocksize_, 0, dev_stream_.get()>>>(
          globals::num_spins,
                  solver->dev_ptr_spin(),
                  dev_direction_.data().get(),
                  dev_magnitude_.data().get(),
                  dev_energy_.data()
  );
}

double CudaRandomAnisotropyHamiltonian::calculate_total_energy() {
  calculate_energies();
  return thrust::reduce(dev_energy_.data(), dev_energy_.data()+dev_energy_.elements());
}