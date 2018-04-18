// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/cuda/cuda_solver.h"

#include <cublas.h>

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/monitor.h"

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

void CudaSolver::compute_fields() {
  using namespace globals;

  for (auto& ham : hamiltonians_) {
    ham->calculate_fields();
  }

  dev_h_.zero();
  for (auto& ham : hamiltonians_) {
    cublasDaxpy(dev_h_.elements(), 1.0, ham->dev_ptr_field(), 1, dev_h_.data(), 1);
  }
}

void CudaSolver::notify_monitors() {
  bool is_device_synchonised = false;
  for (auto& mon : monitors_) {
    if(mon->is_updating(iteration_)){
      if (!is_device_synchonised) {
        sync_device_data();
        is_device_synchonised = true;
      }
      mon->update(this);
    }
  }
}

double *CudaSolver::dev_ptr_spin() {
  return dev_s_.data();
}
