#include <cuda.h>

#include "jams/core/solver.h"
#include "jams/cuda/cuda_common.h"

#include "jams/hamiltonian/uniaxial_microscopic_anisotropy.h"
#include "jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy.h"
#include "jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy_kernel.cuh"

CudaUniaxialMicroscopicHamiltonian::CudaUniaxialMicroscopicHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialMicroscopicHamiltonian(settings, num_spins)
{
  CHECK_CUDA_STATUS(cudaStreamCreate(&dev_stream_));

  dev_blocksize_ = 128;
}

void CudaUniaxialMicroscopicHamiltonian::calculate_fields() {
  cuda_uniaxial_microscopic_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_>>>
            (globals::num_spins, mca_order_.size(), mca_order_.device_data(), mca_value_.device_data(), globals::s.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
