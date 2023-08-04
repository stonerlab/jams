// cuda_landau.cu                                                          -*-C++-*-

#include <jams/hamiltonian/cuda_landau.h>
#include <jams/hamiltonian/cuda_landau_kernel.cuh>
#include <jams/interface/config.h>
#include "jams/core/globals.h"
#include "jams/core/lattice.h"


CudaLandauHamiltonian::CudaLandauHamiltonian(const libconfig::Setting &settings,
                                             const unsigned int size)
    : Hamiltonian(settings, size) {

  landau_A_.resize(globals::num_spins);
  landau_B_.resize(globals::num_spins);
  landau_C_.resize(globals::num_spins);

  for (int i = 0; i < globals::num_spins; ++i) {
    landau_A_(i) = double(settings["A"][globals::lattice->atom_material_id(i)]) * input_energy_unit_conversion_;;
    landau_B_(i) = double(settings["B"][globals::lattice->atom_material_id(i)]) * input_energy_unit_conversion_;;
    landau_C_(i) = double(settings["C"][globals::lattice->atom_material_id(i)]) * input_energy_unit_conversion_;;
  }

}

double CudaLandauHamiltonian::calculate_total_energy(double time) {
  calculate_energies(time);
  double e_total = 0.0;
  for (auto i = 0; i < energy_.size(); ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void CudaLandauHamiltonian::calculate_energies(double time) {
  cuda_landau_energy_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
      (globals::num_spins, globals::s.device_data(), landau_A_.device_data(), landau_B_.device_data(), landau_C_.device_data(), energy_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaLandauHamiltonian::calculate_fields(double time) {
  cuda_landau_field_kernel<<<(globals::num_spins+dev_blocksize_-1)/dev_blocksize_, dev_blocksize_, 0, cuda_stream_.get()>>>
      (globals::num_spins, globals::s.device_data(), landau_A_.device_data(), landau_B_.device_data(), landau_C_.device_data(), field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

