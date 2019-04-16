#include <cuda_runtime_api.h>

#include "jams/core/solver.h"
#include "jams/cuda/cuda_common.h"

#include "jams/hamiltonian/uniaxial_anisotropy.h"
#include "jams/hamiltonian/cuda_uniaxial_anisotropy.h"
#include "jams/hamiltonian/cuda_uniaxial_anisotropy_kernel.cuh"

CudaUniaxialHamiltonian::CudaUniaxialHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialHamiltonian(settings, num_spins)
{
  dev_energy_ = jblib::CudaArray<double, 1>(energy_);
  dev_field_ = jblib::CudaArray<double, 1>(field_);

  dev_power_ = jblib::CudaArray<unsigned, 1>(power_);
  dev_magnitude_ = jblib::CudaArray<double, 1>(magnitude_);

  jblib::Array<double, 2> tmp_axis(axis_.elements(), 3);

  for (auto i = 0; i < axis_.elements(); ++i) {
    for (auto j = 0; j < 3; ++j) {
      tmp_axis(i,j) = axis_[i][j];
    }
  }

  dev_axis_ = jblib::CudaArray<double, 1>(tmp_axis);
}

void CudaUniaxialHamiltonian::calculate_fields() {
  cuda_uniaxial_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
            (globals::num_spins, num_coefficients_, dev_power_.data(), dev_magnitude_.data(), dev_axis_.data(), solver->dev_ptr_spin(), dev_field_.data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
