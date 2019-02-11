#include <cuda.h>

#include "jams/core/solver.h"
#include "jams/cuda/cuda_common.h"

#include "jams/hamiltonian/uniaxial_microscopic_anisotropy.h"
#include "jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy.h"
#include "jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy_kernel.cuh"

CudaUniaxialMicroscopicHamiltonian::CudaUniaxialMicroscopicHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialMicroscopicHamiltonian(settings, num_spins)
{
  dev_energy_ = jblib::CudaArray<double, 1>(energy_);
  dev_field_ = jblib::CudaArray<double, 1>(field_);

  jblib::Array<int, 1> tmp_mca_order(mca_order_.size());
  for (int i = 0; i < mca_order_.size(); ++i) {
      tmp_mca_order[i] = mca_order_[i];
  }

  dev_mca_order_ = jblib::CudaArray<int, 1>(tmp_mca_order);

  jblib::Array<double, 1> tmp_mca_value(mca_order_.size() * num_spins);

  for (int i = 0; i < num_spins; ++i) {
      for (int j = 0; j < mca_order_.size(); ++j) {
          tmp_mca_value[ mca_order_.size() * i + j] = mca_value_[j](i);
      }
  }
  dev_mca_value_ = tmp_mca_value;

  CHECK_CUDA_STATUS(cudaStreamCreate(&dev_stream_));

  dev_blocksize_ = 128;
}

void CudaUniaxialMicroscopicHamiltonian::calculate_fields() {
  cuda_uniaxial_microscopic_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_>>>
            (globals::num_spins, mca_order_.size(), dev_mca_order_.data(), dev_mca_value_.data(), solver->dev_ptr_spin(), dev_field_.data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
