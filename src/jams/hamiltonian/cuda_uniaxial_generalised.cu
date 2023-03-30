//
// Created by Sean Stansill [ll14s26s] on 30/03/2023.
//

#include <jams/hamiltonian/cuda_uniaxial_generalised.h>
#include <jams/hamiltonian/cuda_uniaxial_generalised_kernel.cuh>
#include <jams/hamiltonian/uniaxial_generalised.h>

#include <jams/cuda/cuda_common.h>
#include <jams/core/globals.h>


CudaUniaxialHamiltonian::CudaUniaxialGeneralisedHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialGeneralisedHamiltonian(settings, num_spins)
{
}

double CudaUniaxialGeneralisedHamiltonian::calculate_total_energy(double time) {
    calculate_energies(time);
    double e_total = 0.0;
    for (auto i = 0; i < energy_.size(); ++i) {
        e_total += energy_(i);
    }
    return e_total;
}

void CudaUniaxialGeneralisedHamiltonian::calculate_energies(double time) {
    cuda_uniaxial_generalised_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
            (globals::num_spins, magnitude_.device_data(), axis1_.device_data(), axis2_.device_data(), axis3_.device_data(), globals::s.device_data(), energy_.device_data(),
             a1_, a2_, a3_, a4_, a5_, a6_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaUniaxialGeneralisedHamiltonian::calculate_fields(double time) {
    cuda_uniaxial_generalised_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
            (globals::num_spins, magnitude_.device_data(), axis1_.device_data(), axis2_.device_data(), axis3_.device_data(), globals::s.device_data(), field_.device_data(),
             a1_, a2_, a3_, a4_, a5_, a6_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
