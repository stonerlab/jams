#include <jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy.h>
#include <jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy_kernel.cuh>

#include <jams/hamiltonian/uniaxial_microscopic_anisotropy.h>
#include <jams/cuda/cuda_common.h>
#include <jams/core/globals.h>

CudaUniaxialMicroscopicAnisotropyHamiltonian::CudaUniaxialMicroscopicAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialMicroscopicAnisotropyHamiltonian(settings, num_spins)
{
  dev_blocksize_ = 128;
}

void CudaUniaxialMicroscopicAnisotropyHamiltonian::calculate_fields(double time) {
  cuda_uniaxial_microscopic_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
            (globals::num_spins, mca_order_.size(), mca_order_.device_data(), mca_value_.device_data(), globals::s.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
