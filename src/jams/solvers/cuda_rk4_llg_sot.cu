#include <jams/solvers/cuda_rk4_llg_sot.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>
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

  // We assume the sample is a metal/magnet bilayer. In the experiment a
  // total charge current (Jc) is measured in amps.We need to use the
  // sample dimensions, length (l), width (w) thickness (t), to convert to the
  // total spin current through the interface (Js).
  //
  //               magnet
  //             +--------------------------------+
  //            /                                /|
  //           /                                / |
  //          +--------------------------------+ /|
  //          |            Js                  |/ |
  //        ↑ +-----------↑↑↑↑-----------------+ / ↗
  //        t |             →→ Jc →→            |/ w
  //        ↓ +--------------------------------+ ↙
  //           metal         ← l →
  //
  // Jc is the charge passing through an area per unit time. In the experiment
  // the area is the plane of the metal normal to the current direction.
  // Assuming the current is applied in-plane, this is then the area t x w.
  //
  // The spin current Js flows perpendicular to Jc, from the metal into the
  // magnet. Through the spin Hall effect, each electron converts to (ℏ/2) θ_SH
  // units of spin angular moment, with θ_SH the spin Hall angle. There is a
  // subtlety here because Js is the spin angular momentum through an area per
  // unit time, but it is NOT the same area as the charge current passes through.
  // The spin current passes through the interface plane, which has the area
  // l x w.
  //
  // So the total spin current flowing through the interface from electrons in
  // the metal is
  //
  // Js = (Jc / e) (ℏ/2) θ_SH (l / t)
  //
  // which has units of spin angular momentum per unit time
  //
  // The spin current density is then
  //
  // js = Js / (l w)
  //
  // Now things become a little tricky. We define the cross sectional area
  // that each atom has (a_atom) with respect to the Wigner-Seitz volume (v_ws)
  // as
  //
  // a_atom = (v_ws)^(2/3)
  //
  // So the spin current felt by each atom in the magnet is
  //
  // js_atom = js * (v_ws)^(2/3)
  //         = (v_ws)^(2/3) (Jc / tw) (ℏ/2e) θ_SH
  //
  // This then converts to an effective field using Beff = (1/μ) * js_atom.

  auto charge_current_density = jams::config_required<double>(settings, "charge_current_density");

  // We need to convert the charge current density into internal units. An ampere is 1 C/s, unit of charge 'e'
  // is unchanged in JAMS but the time should be in picoseconds. The sheet area (t.w) needs to be converted
  // from meters^2 to nanometers^2.
  charge_current_density = charge_current_density / (kMeterToNanometer * kMeterToNanometer * kSecondToPicosecond);

  double volume_per_atom = volume(globals::lattice->get_unitcell()) / double(globals::lattice->num_motif_atoms());

  sot_coefficient_.resize(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    sot_coefficient_(i) = kHBarIU * spin_hall_angle * charge_current_density *
                          std::pow(volume_per_atom, 2.0/3.0) / (2 * kElementaryCharge);
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
