// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/cuda/cuda_solver.h"

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/monitor.h"
#include "jams/cuda/cuda_common.h"

using namespace std;

void CudaSolver::initialize(const libconfig::Setting& settings) {
  using namespace globals;

  Solver::initialize(settings);

  cout << "\ninitializing CUDA base solver\n";
  cout << "  initialising CUDA streams\n";

  is_cuda_solver_ = true;

  cout << "\n";
}

void CudaSolver::compute_fields() {
  using namespace globals;

  if (hamiltonians_.empty()) return;

  for (auto& hh : hamiltonians_) {
    hh->calculate_fields();
  }

  cudaMemcpy(globals::h.device_data(),hamiltonians_[0]->dev_ptr_field(), globals::num_spins3*sizeof(double) ,cudaMemcpyDeviceToDevice);

  if (hamiltonians_.size() == 1) return;

  for (auto i = 1; i < hamiltonians_.size(); ++i) {
    CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(),globals::h.elements(), &kOne, hamiltonians_[i]->dev_ptr_field(), 1, globals::h.device_data(), 1));
  }

}

