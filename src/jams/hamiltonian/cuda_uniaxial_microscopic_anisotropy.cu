#include <jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy.h>

#include <jams/hamiltonian/uniaxial_microscopic_anisotropy.h>
#include <jams/cuda/cuda_common.h>
#include <jams/core/globals.h>

#include "jams/cuda/cuda_legendre.h"

__global__ void cuda_uniaxial_microscopic_energy_kernel(const int num_spins, const int num_mca, const int * mca_order,
  const jams::Real * mca_value, const jams::RealHi * dev_s, jams::Real * dev_e) {
    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= num_spins) return;

    const jams::Real sz = static_cast<jams::Real>(dev_s[3*idx+2]);
    jams::Real energy = 0.0;
    for (int n = 0; n < num_mca; ++n) {
        energy += mca_value[num_spins * n + idx] * cuda_legendre_poly(sz, mca_order[n]);
    }
    dev_e[idx] =  energy;
}

__global__ void cuda_uniaxial_microscopic_field_kernel(const int num_spins, const int num_mca, const int * mca_order,
  const jams::Real * mca_value, const jams::RealHi * dev_s, jams::Real * dev_h) {
    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= num_spins) return;
    const jams::Real sz = static_cast<jams::Real>(dev_s[3*idx+2]);
    jams::Real hz = 0.0;
    for (int n = 0; n < num_mca; ++n) {
        hz += -mca_value[num_spins * n + idx] * cuda_legendre_dpoly(sz, mca_order[n]);
    }
    dev_h[3 * idx + 2] =  hz;
}


CudaUniaxialMicroscopicAnisotropyHamiltonian::CudaUniaxialMicroscopicAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialMicroscopicAnisotropyHamiltonian(settings, num_spins)
{
  dev_blocksize_ = 128;
}

void CudaUniaxialMicroscopicAnisotropyHamiltonian::calculate_fields(jams::Real time) {
  cuda_uniaxial_microscopic_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
            (globals::num_spins, mca_order_.size(), mca_order_.device_data(), mca_value_.device_data(), globals::s.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaUniaxialMicroscopicAnisotropyHamiltonian::calculate_energies(jams::Real time)
{
    cuda_uniaxial_microscopic_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
            (globals::num_spins, mca_order_.size(), mca_order_.device_data(), mca_value_.device_data(), globals::s.device_data(), energy_.device_data());
}