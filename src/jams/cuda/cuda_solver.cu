// Copyright 2014 Joseph Barker. All rights reserved.
#include "jams/cuda/cuda_solver.h"

#include <jams/common.h>
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include <jams/cuda/cuda_common.h>

#include <cublas_v2.h>

#include "cuda_array_kernels.h"


void CudaSolver::compute_fields() {
  if (hamiltonians_.empty()) return;

  auto master_stream = jams::instance().cuda_master_stream().get();

#if DO_MIXED_PRECISION
  const auto& spins = refresh_field_spin_array_async();
#else
  const auto& spins = globals::s;
#endif

  for (auto& hh : hamiltonians_) {
#if DO_MIXED_PRECISION
    wait_on_field_spin_cache_event(hh->get_stream());
#else
    wait_on_spin_barrier_event(hh->get_stream());
#endif
    hh->calculate_fields(this->time(), spins);
    hh->record_done();
  }

  for (auto& hh : hamiltonians_) {
    hh->wait_on(master_stream);
  }

  const int num_input_arrays = static_cast<int>(hamiltonians_.size());
  const int num_elements = globals::h.elements(); // == globals::num_spins3

  if (num_input_arrays == 1) {
    CHECK_CUDA_STATUS(cudaMemcpyAsync(
        globals::h.mutable_device_data(),
        hamiltonians_[0]->dev_ptr_field(),
        globals::h.bytes(),
        cudaMemcpyDeviceToDevice,
        master_stream));
    return;
  }

  if (dev_field_ptrs_ == nullptr) {
    // Collect device pointers on host
    std::vector<jams::Real*> h_ptrs(num_input_arrays);
    for (int i = 0; i < num_input_arrays; ++i) {
      h_ptrs[i] = hamiltonians_[i]->dev_ptr_field();
    }

    // Copy pointer array to device (cache this if topology is fixed)
    cudaMallocAsync(&dev_field_ptrs_, num_input_arrays * sizeof(jams::Real*), master_stream);
    cudaMemcpyAsync(dev_field_ptrs_, h_ptrs.data(),
               num_input_arrays * sizeof(jams::Real*),
               cudaMemcpyHostToDevice,
               master_stream);
  }

  cuda_array_sum_across(
      num_input_arrays,
      num_elements,
      dev_field_ptrs_,
      globals::h.mutable_device_data(),
      master_stream);
}

const jams::MultiArray<jams::Real, 2>& CudaSolver::spin_array_for_fields() {
#if DO_MIXED_PRECISION
  const auto& spins = refresh_field_spin_array_async();
  cudaEventSynchronize(field_spin_cache_event_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  return spins;
#else
  synchronize_on_spin_barrier_event();
  return globals::s;
#endif
}

#if DO_MIXED_PRECISION
const jams::MultiArray<jams::Real, 2>& CudaSolver::refresh_field_spin_array_async() {
  if (field_spin_array_.elements() != globals::s.elements()) {
    field_spin_array_.resize(globals::s.extent(0), globals::s.extent(1));
  }

  auto& master_stream = jams::instance().cuda_master_stream().get();
  wait_on_spin_barrier_event(master_stream);
  cuda_array_double_to_float(
      globals::s.elements(),
      globals::s.device_data(),
      field_spin_array_.mutable_device_data(),
      master_stream);

  if (!field_spin_cache_event_) create_field_spin_cache_event();
  cudaEventRecord(field_spin_cache_event_, master_stream);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  return field_spin_array_;
}
#endif
