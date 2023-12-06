#include <jams/hamiltonian/cuda_uniaxial_anisotropy.h>
#include <jams/hamiltonian/cuda_uniaxial_anisotropy_kernel.cuh>
#include <jams/hamiltonian/uniaxial_anisotropy.h>

#include <jams/core/globals.h>


CudaUniaxialAnisotropyHamiltonian::CudaUniaxialAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int num_spins)
        : UniaxialAnisotropyHamiltonian(settings, num_spins)
{
}

double CudaUniaxialAnisotropyHamiltonian::calculate_total_energy(double time) {
  calculate_energies(time);
  double e_total = 0.0;
  for (auto i = 0; i < energy_.size(); ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void CudaUniaxialAnisotropyHamiltonian::calculate_energies(double time) {
  cuda_uniaxial_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
  (globals::num_spins, power_, magnitude_.device_data(), axis_.device_data(), globals::s.device_data(), energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaUniaxialAnisotropyHamiltonian::calculate_fields(double time) {
  cuda_uniaxial_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, dev_stream_.get()>>>
            (globals::num_spins, power_, magnitude_.device_data(), axis_.device_data(), globals::s.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}
