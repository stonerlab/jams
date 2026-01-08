// cuda_landau.cu                                                          -*-C++-*-

#include <jams/hamiltonian/cuda_landau.h>
#include <jams/interface/config.h>
#include "jams/core/globals.h"
#include "jams/core/lattice.h"

__global__
void cuda_landau_field_kernel(
    const int num_spins,
    const double * spins,
    const jams::Real *A,
    const jams::Real *B,
    const jams::Real *C,
    jams::Real * fields)
{
  const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  const jams::Real s[3] = {static_cast<jams::Real>(spins[base + 0]), static_cast<jams::Real>(spins[base + 1]), static_cast<jams::Real>(spins[base + 2])};
  const jams::Real s2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];

  jams::Real h[3] = {0, 0, 0};

  for (auto n = 0; n < 3; ++n) {
    h[n] += -static_cast<jams::Real>(2.0) * A[idx] * s[n];
  }

  for (auto n = 0; n < 3; ++n) {
    h[n] += -static_cast<jams::Real>(4.0) * B[idx] * s[n] * s2;
  }


  for (auto n = 0; n < 3; ++n) {
    h[n] += -static_cast<jams::Real>(6.0) * C[idx] * s[n] * s2 * s2;
  }

  for (auto n = 0; n < 3; ++n) {
    fields[base + n] = h[n];
  }
}

__global__
void cuda_landau_energy_kernel(
    const int num_spins,
    const double * spins,
    const jams::Real *A,
    const jams::Real *B,
    const jams::Real *C,
    jams::Real * energies)
{
  const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  const jams::Real s[3] = {static_cast<jams::Real>(spins[base + 0]), static_cast<jams::Real>(spins[base + 1]), static_cast<jams::Real>(spins[base + 2])};

  const jams::Real s2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];

  energies[idx] = A[idx] * s2 + B[idx] * s2 * s2 + C[idx] * s2 * s2 * s2;
}

CudaLandauHamiltonian::CudaLandauHamiltonian(const libconfig::Setting &settings,
                                             const unsigned int size)
    : Hamiltonian(settings, size) {

  landau_A_.resize(globals::num_spins);
  landau_B_.resize(globals::num_spins);
  landau_C_.resize(globals::num_spins);

  for (int i = 0; i < globals::num_spins; ++i) {
    landau_A_(i) = jams::Real(settings["A"][globals::lattice->lattice_site_material_id(i)]) * input_energy_unit_conversion_;;
    landau_B_(i) = jams::Real(settings["B"][globals::lattice->lattice_site_material_id(i)]) * input_energy_unit_conversion_;;
    landau_C_(i) = jams::Real(settings["C"][globals::lattice->lattice_site_material_id(i)]) * input_energy_unit_conversion_;;
  }

}

jams::Real CudaLandauHamiltonian::calculate_total_energy(jams::Real time) {
  calculate_energies(time);
  jams::Real e_total = 0.0;
  for (auto i = 0; i < energy_.size(); ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void CudaLandauHamiltonian::calculate_energies(jams::Real time) {
  cuda_landau_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
      (globals::num_spins, globals::s.device_data(), landau_A_.device_data(), landau_B_.device_data(), landau_C_.device_data(), energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaLandauHamiltonian::calculate_fields(jams::Real time) {
  cuda_landau_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
      (globals::num_spins, globals::s.device_data(), landau_A_.device_data(), landau_B_.device_data(), landau_C_.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

