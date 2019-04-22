// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/cuda/cuda_solver.h"

#include <cublas_v2.h>

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

  CHECK_CUBLAS_STATUS(cublasCreate(&cublas_handle_));

  cout << "\n";
}

void CudaSolver::compute_fields() {
  using namespace globals;

  for (auto& ham : hamiltonians_) {
    ham->calculate_fields();
  }

  globals::h.zero();
  for (auto& ham : hamiltonians_) {
    const double alpha = 1.0;
    CHECK_CUBLAS_STATUS(cublasDaxpy(cublas_handle_,globals::h.elements(), &alpha, ham->dev_ptr_field(), 1, globals::h.device_data(), 1));
  }
}

