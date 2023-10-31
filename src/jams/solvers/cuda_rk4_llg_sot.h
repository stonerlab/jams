#ifndef JAMS_SOLVER_CUDA_RK4_LLG_SOT_H
#define JAMS_SOLVER_CUDA_RK4_LLG_SOT_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/containers/multiarray.h"
#include "jams/solvers/cuda_rk4_base.h"

///
/// RK4 LLG solver including spin orbit torque
///
/// Solves the equation
///
/// dSᵢ / dt = −γᵢ(Sᵢ x Hᵢ + αᵢ Sᵢ x (Sᵢ x Hᵢ) + (βᵢ / μᵢ) * (Sᵢ x (Sᵢ x σ)))
///
/// where βᵢ = v^{2/3} (Jc / (t x w)) (ℏ/2e) θ_SH is the spin current per spin
/// due to spin orbit torques from a adjacent metal thin film of thickness, t,
/// and width, w, with an electronic charge current Jc measured in amps and
/// spin Hall angle θ_SH. σ is the electron spin polarisation vector.
///
/// Settings
/// --------
/// The Solver settings are:
///
/// module       : "llg-sot-rk4-gpu"
///
/// spin_polarisation      : (list) vector of the spin polarisation
///
/// spin_hall_angle        : (float) spin Hall angle
///
/// charge_current_density : (float) value of the sheet charge current
///                          Jc / (t x w) in units A/m^2.
///
/// Example
/// -------
///
/// solver : {
///   module = "llg-sot-rk4-gpu";
///   t_step = 1e-15;
///   t_max  = 1e-9;
///   spin_polarisation = [0.0, -1.0, 0.0];
///   spin_hall_angle = 0.1;
///
///   // This should be the charge current in Amps, divided by (t x w) which
///   // is the metal thickness and width of the sample.
///   // For Jc = 1mA, t = 1nm, w = 1μm we get 1e12 A/m^2 which in JAMS units is
///   charge_current_density = 1.0e12;
/// };
///

class CudaRK4LLGSOTSolver : public CudaRK4BaseSolver {
public:
    explicit CudaRK4LLGSOTSolver(const libconfig::Setting &settings);

    std::string name() const override { return "llg-sot-rk4-gpu"; }

    void function_kernel(jams::MultiArray<double, 2>& spins, jams::MultiArray<double, 2>& k) override;
    void post_step(jams::MultiArray<double, 2>& spins) override;

private:
    jams::MultiArray<double, 2> spin_polarisation_;
    jams::MultiArray<double, 1> sot_coefficient_;
};

#endif

#endif // JAMS_SOLVER_CUDA_RK4_LLG_SOT_H

