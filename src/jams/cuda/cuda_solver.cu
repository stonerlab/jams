// Copyright 2014 Joseph Barker. All rights reserved.

#include <cublas.h>

#include "cuda_solver.h"
#include "cuda_solver_kernels.h"

#include "jams/helpers/consts.h"
#include "cuda_defs.h"
#include "cuda_sparsematrix.h"
#include "jams/helpers/exception.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/solver.h"
#include "jams/core/thermostat.h"
#include "jams/helpers/utils.h"
#include "jams/solvers/cuda_llg_heun.h"
#include "jams/solvers/cpu_llg_heun.h"
#include "jams/solvers/cpu_monte_carlo_metropolis.h"

#include "jams/cuda/wrappers/stream.h"

using namespace std;

void CudaSolver::sync_device_data() {
  dev_s_.copy_to_host_array(globals::s);
  dev_h_.copy_to_host_array(globals::h);
  dev_ds_dt_.copy_to_host_array(globals::ds_dt);
}

void CudaSolver::initialize(const libconfig::Setting& settings) {
  using namespace globals;

  Solver::initialize(settings);

  cout << "\ninitializing CUDA base solver\n";
  cout << "  initialising CUDA streams\n";

  is_cuda_solver_ = true;

//-----------------------------------------------------------------------------
// Transfer the the other arrays to the device
//-----------------------------------------------------------------------------

  cout << "  transfering array data to device\n";
  jblib::Array<double, 2> zero(num_spins, 3, 0.0);

  // spin arrays
  dev_s_        = jblib::CudaArray<double, 1>(s);
  dev_s_old_    = jblib::CudaArray<double, 1>(s);
  dev_ds_dt_    = jblib::CudaArray<double, 1>(zero);

  // field array
  dev_h_        = jblib::CudaArray<double, 1>(zero);

  // materials array
  jblib::Array<double, 2> mat(num_spins, 3);

  dev_gyro_      = jblib::CudaArray<double, 1>(gyro);
  dev_alpha_     = jblib::CudaArray<double, 1>(alpha);


  cout << "\n";
}

void CudaSolver::run() {
}

void CudaSolver::compute_fields() {
  using namespace globals;

  for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    (*it)->calculate_fields();
  }

  // zero the field array
  if (cudaMemsetAsync(dev_h_.data(), 0.0, num_spins3*sizeof(double), dev_stream_.get()) != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  const double alpha = 1.0;
  for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    cublasDaxpy(globals::num_spins3, alpha, (*it)->dev_ptr_field(), 1, dev_h_.data(), 1);
  }
}

CudaSolver::~CudaSolver() {
}
