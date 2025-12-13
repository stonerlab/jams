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

  for (auto& hh : hamiltonians_) {
    hh->calculate_fields(this->time());
  }

  // cudaMemcpy(globals::h.device_data(),hamiltonians_[0]->dev_ptr_field(), globals::num_spins3*sizeof(double) ,cudaMemcpyDeviceToDevice);
  //
  // if (hamiltonians_.size() == 1) return;
  //
  // for (auto i = 1; i < hamiltonians_.size(); ++i) {
  //   CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(),globals::h.elements(), &kOne, hamiltonians_[i]->dev_ptr_field(), 1, globals::h.device_data(), 1));
  // }

  const int num_input_arrays = static_cast<int>(hamiltonians_.size());
  const int num_elements = globals::h.elements(); // == globals::num_spins3


  if (dev_field_ptrs_ == nullptr) {
    // Collect device pointers on host
    std::vector<double*> h_ptrs(num_input_arrays);
    for (int i = 0; i < num_input_arrays; ++i) {
      h_ptrs[i] = hamiltonians_[i]->dev_ptr_field();
    }

    // Copy pointer array to device (cache this if topology is fixed)
    cudaMalloc(&dev_field_ptrs_, num_input_arrays * sizeof(double*));
    cudaMemcpy(dev_field_ptrs_, h_ptrs.data(),
               num_input_arrays * sizeof(double*),
               cudaMemcpyHostToDevice);
  }

  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;
  cuda_array_sum_across<<<grid_size, block_size>>>(num_input_arrays, num_elements, dev_field_ptrs_, globals::h.device_data());

}

