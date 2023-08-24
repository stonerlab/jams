#include <jams/solvers/cuda_rk4_llg_sot.h>
#include <jams/core/globals.h>
#include <jams/interface/config.h>
#include <jams/solvers/cuda_rk4_llg_sot_kernel.cuh>
#include "jams/cuda/cuda_spin_ops.h"


CudaRK4LLGSOTSolver::CudaRK4LLGSOTSolver(const libconfig::Setting &settings)
    : CudaRK4BaseSolver(settings) {


  auto spin_polarisation = jams::config_required<Vec3>(settings, "spin_polarisation");
  spin_polarisation_.resize(globals::num_spins, 3);
  // Assume the charge current is homogeneous. The current here is a dimensionful
  // vector in
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto n = 0; n < 3; ++n) {
      spin_polarisation_(i, n) = spin_polarisation[n];
    }
  }

  auto spin_hall_angle = jams::config_required<double>(settings, "spin_hall_angle");
  auto charge_current = jams::config_required<double>(settings, "charge_current_density");

  // WHAT TO DO FOR Ms * t?

  sot_coefficient_.resize(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    sot_coefficient_(i) = kHBarIU * spin_hall_angle * charge_current
                          / (2 * kElementaryCharge);
  }
}

void CudaRK4LLGSOTSolver::function_kernel(jams::MultiArray<double, 2> &spins,
                                          jams::MultiArray<double, 2> &k) {

  jams::normalise_spins_cuda(spins);

  compute_fields();

  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  // using default stream blocks all streams until complete to force synchronisation
  cuda_rk4_llg_sot_kernel<<<grid_size, block_size>>>
      (spins.device_data(), k.device_data(),
       globals::h.device_data(), spin_polarisation_.device_data(),
       sot_coefficient_.device_data(), thermostat_->device_data(),
       globals::gyro.device_data(), globals::mus.device_data(),
       globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}


void CudaRK4LLGSOTSolver::post_step(jams::MultiArray<double, 2> &spins) {

}
