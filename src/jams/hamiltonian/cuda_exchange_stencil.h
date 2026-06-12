// Copyright 2026 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_CUDA_EXCHANGE_STENCIL_H
#define JAMS_HAMILTONIAN_CUDA_EXCHANGE_STENCIL_H

#include <jams/containers/block_sparse_interaction_matrix.h>
#include <jams/hamiltonian/exchange_stencil.h>

#if HAS_CUDA

class CudaExchangeStencilHamiltonian : public ExchangeStencilHamiltonian {
public:
  CudaExchangeStencilHamiltonian(const libconfig::Setting& settings, unsigned int size);

  jams::Real calculate_total_energy(jams::Real time, const SpinArray& spins) override;
  void calculate_energies(jams::Real time, const SpinArray& spins) override;
  void calculate_fields(jams::Real time, const SpinArray& spins) override;

private:
  void upload_stencil_to_device();

  jams::MultiArray<int, 1> device_entry_offsets_;
  jams::MultiArray<int, 1> device_entry_dx_;
  jams::MultiArray<int, 1> device_entry_dy_;
  jams::MultiArray<int, 1> device_entry_dz_;
  jams::MultiArray<int, 1> device_target_basis_;
  jams::MultiArray<int, 1> device_group_offsets_;
  jams::MultiArray<int, 1> device_group_entry_offsets_;
  jams::MultiArray<int, 1> device_group_translation_ids_;
  jams::MultiArray<int, 1> device_group_dx_;
  jams::MultiArray<int, 1> device_group_dy_;
  jams::MultiArray<int, 1> device_group_dz_;
  jams::MultiArray<int, 1> device_translation_target_cells_;
  jams::MultiArray<jams::Real, 1> device_values_;
  int num_cell_translations_ = 0;
  int num_stencil_entries_ = 0;
  int num_stencil_groups_ = 0;
  int field_kernel_block_size_ = 128;
  int components_per_entry_ = 1;
  bool use_direct_periodic_kernel_ = false;
  bool fully_periodic_ = false;
  jams::InteractionTensorStorage tensor_storage_ = jams::InteractionTensorStorage::Isotropic;
};

#endif  // HAS_CUDA

#endif  // JAMS_HAMILTONIAN_CUDA_EXCHANGE_STENCIL_H
