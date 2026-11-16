#include "jams/solvers/cuda_rk4_llg_sot.h"

#include <cmath>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/cuda/cuda_common.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/cuda/cuda_spin_ops.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"
#include "jams/interface/config.h"

namespace {

__global__ void cuda_rk4_llg_sot_kernel(const double* spins, double* slopes,
                                        const jams::Real* fields, const double* polarisation_dev,
                                        const double* sot_dev, const jams::Real* noise,
                                        const jams::Real* gyro_eff,
                                        const jams::Real* inverse_moments,
                                        const jams::Real* damping, const unsigned num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (spin_index >= num_spins) return;
  const unsigned int offset = 3 * spin_index;

  jams::Real h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] =
        ((fields[offset + n] * inverse_moments[spin_index]) + noise[offset + n]);
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = spins[offset + n];
  }

  double j[3];
  for (auto n = 0; n < 3; ++n) {
    j[n] = polarisation_dev[offset + n];
  }

  double sxh[3] = {(s[1] * h[2] - s[2] * h[1]), (s[2] * h[0] - s[0] * h[2]),
                   (s[0] * h[1] - s[1] * h[0])};

  double sxsxh[3] = {(s[1] * sxh[2] - s[2] * sxh[1]), (s[2] * sxh[0] - s[0] * sxh[2]),
                     (s[0] * sxh[1] - s[1] * sxh[0])};

  // using vector triple product
  // s x s x j = (s . j) s - (s . s) j
  // and accepting that s may not be a unit vector

  double s_dot_j = dot(s, j);
  double s_dot_s = dot(s, s);

  double sxsxj[3] = {s_dot_j * s[0] - s_dot_s * j[0], s_dot_j * s[1] - s_dot_s * j[1],
                     s_dot_j * s[2] - s_dot_s * j[2]};

  for (auto n = 0; n < 3; ++n) {
    slopes[offset + n] =
        -gyro_eff[spin_index] *
        (sxh[n] + damping[spin_index] * sxsxh[n] +
         (sot_dev[spin_index] * inverse_moments[spin_index]) * sxsxj[n]);
  }
}

}  // namespace

CudaRK4LLGSOTSolver::CudaRK4LLGSOTSolver(const libconfig::Setting& settings)
    : CudaRK4BaseSolver(settings) {
  auto spin_polarisation =
      jams::config_required<jams::Vec<double, 3>>(settings, "spin_polarisation");
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

  // We need to convert the charge current density into internal units. An ampere is 1 C/s, unit of
  // charge 'e' is unchanged in JAMS but the time should be in picoseconds. The sheet area (t.w)
  // needs to be converted from meters^2 to nanometers^2.
  charge_current_density =
      charge_current_density / (kMeterToNanometer * kMeterToNanometer * kSecondToPicosecond);

  double volume_per_atom = pow3(globals::lattice->parameter() * kMeterToNanometer) *
                           volume(globals::lattice->get_unitcell()) /
                           double(globals::lattice->num_basis_sites());

  sot_coefficient_.resize(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    sot_coefficient_(i) = kHBarIU * spin_hall_angle * charge_current_density *
                          std::pow(volume_per_atom, 2.0 / 3.0) / (2 * kElementaryCharge);
  }
}

void CudaRK4LLGSOTSolver::pre_stage(jams::MultiArray<double, 2>& spins, cudaStream_t stream) {
  jams::normalise_spins_cuda(spins, stream);
}

void CudaRK4LLGSOTSolver::function_kernel(jams::MultiArray<double, 2>& spins,
                                          jams::MultiArray<double, 2>& k, cudaStream_t stream) {
  const dim3 block_size = {64, 1, 1};
  const auto grid_size =
      cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_rk4_llg_sot_kernel<<<grid_size, block_size, 0, stream>>>(
      spins.device_data(), k.mutable_device_data(), globals::h.device_data(),
      spin_polarisation_.device_data(), sot_coefficient_.device_data(), thermostat_->device_data(),
      gyro_eff_.device_data(), globals::inv_mus.device_data(), globals::alpha.device_data(),
      globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

void CudaRK4LLGSOTSolver::post_step(jams::MultiArray<double, 2>& spins, cudaStream_t stream) {
  jams::normalise_spins_cuda(spins, stream);
}
