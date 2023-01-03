#include <jams/hamiltonian/cuda_cubic_anisotropy.h>
#include <jams/hamiltonian/cuda_cubic_anisotropy_kernel.cuh>

#include <jams/cuda/cuda_common.h>
#include <jams/core/globals.h>
#include <jams/hamiltonian/cubic_anisotropy.h>


CudaCubicHamiltonian::CudaCubicHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
    : CubicHamiltonian(settings, num_spins)
{}

void CudaCubicHamiltonian::calculate_fields(double time) {
  cuda_cubic_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
                                                                                                     (globals::num_spins, num_coefficients_, order_.device_data(), magnitude_.device_data(),
                                                                                                         reinterpret_cast<const double*>(axis1_.device_data()),
                                                                                                         reinterpret_cast<const double*>(axis2_.device_data()),
                                                                                                         reinterpret_cast<const double*>(axis3_.device_data()),
                                                                                                         globals::s.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
