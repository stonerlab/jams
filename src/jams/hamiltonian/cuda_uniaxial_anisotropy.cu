#include <cuda_runtime_api.h>

#include "jams/core/solver.h"
#include "jams/cuda/cuda_common.h"

#include "jams/hamiltonian/uniaxial_anisotropy.h"
#include "jams/hamiltonian/cuda_uniaxial_anisotropy.h"
#include "jams/hamiltonian/cuda_uniaxial_anisotropy_kernel.cuh"

CudaUniaxialHamiltonian::CudaUniaxialHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialHamiltonian(settings, num_spins)
{
    helicity_internal_.resize(globals::num_spins);
    helicity_entropy_.resize(globals::num_spins);
}

double CudaUniaxialHamiltonian::calculate_total_energy() {
  calculate_energies();
  double e_total = 0.0;
  for (auto i = 0; i < energy_.size(); ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void CudaUniaxialHamiltonian::calculate_energies() {
  cuda_uniaxial_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
  (globals::num_spins, power_, magnitude_.device_data(), axis_.device_data(), globals::s.device_data(), energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaUniaxialHamiltonian::calculate_fields() {
  cuda_uniaxial_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
            (globals::num_spins, power_, magnitude_.device_data(), axis_.device_data(), globals::s.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaUniaxialHamiltonian::calculate_internal_energy_differences() {
    cuda_uniaxial_helicity_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
            (globals::num_spins, power_, magnitude_.device_data(), axis_.device_data(), globals::s.device_data(), helicity_internal_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaUniaxialHamiltonian::calculate_entropies() {
    cuda_uniaxial_entropy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
            (globals::num_spins, power_, magnitude_.device_data(), axis_.device_data(), globals::s.device_data(), helicity_entropy_.device_data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

double CudaUniaxialHamiltonian::calculate_total_entropy() {
    calculate_internal_energy_differences();
    double TS_total = 0.0;
    for (auto i = 0; i < energy_.size(); ++i) {
        TS_total += helicity_entropy_(i);
    }
    return TS_total*TS_total;
}

double CudaUniaxialHamiltonian::calculate_total_internal_energy_difference() {
    calculate_entropies();
    double dU_total = 0.0;
    for (auto i = 0; i < energy_.size(); ++i) {
        dU_total += helicity_internal_(i);
    }
    return dU_total;
}
