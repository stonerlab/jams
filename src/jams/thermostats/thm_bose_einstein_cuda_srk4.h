// thm_bose_einstein_cuda_srk4.h                                       -*-C++-*-

#ifndef INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4
#define INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4

#if HAS_CUDA

#include "jams/core/thermostat.h"
#include "jams/containers/multiarray.h"
#include "jams/cuda/cuda_stream.h"

/// ******************************** WARNING ***********************************
/// This thermostat is currently broken. Do not use for production work.
/// ****************************************************************************

/// This implements a thermostat which has the statistics of a Bose-Einstein
/// distribution which has the correlations
///
/// \f[
///       \langle \xi_{i\alpha}(t) \rangle = 0
/// \f]
///
/// \f[
///     \langle \xi_{i\alpha} \xi_{j\beta} \rangle_{\omega} =
///         \delta_{ij}\delta_{\alpha\beta}
///         \frac{2\lambda_i \kB T}{\gamma\mu}
///         \frac{\hbar\omega}{\exp{\hbar\omega/k_BT} - 1}
/// \f]
///
/// where i,j are different spins, alpha, beta are Cartesian components, gamma
/// is the gyromagnetic ratio, mu is the magnetic moments of a spin and lambda_i
/// is the coupling between the spin and the heat bath.
///
/// Coloured noise with the target spectrum is created using stochastic
/// differential equations following the method in Savin, Phys. Rev. B 86,
/// 064305 (2012), https://dx.doi.org/10.1103/physrevb.86.064305. We ignore the
/// zero point contribution (see below) and solve two second order stochastic
/// differential equations to produce the noise (see Eqs. (29)--(31)).
///
/// We solve the stochastic differential equations using a fourth order
/// stochastic Runge-Kutta (SRK4) derived in Honeycutt, Phys. Rev. A 45, 604
/// (1992) https://dx.doi.org/10.1103/physreva.45.604. See
/// thm_bose_einstein_cuda_srk4_kernel.cuh for details of the integration.
///
/// This is essentially a "quantum" thermostat but without the zero-point term
/// (+ hbar omega / 2). Including this term means the power of the spectrum
/// increases linearly with frequency and the system is always unstable. There
/// may also be physics motivation not to include the zero point term in a
/// Heisenberg model (fixed spin length) but this is a problem of current
/// research.

namespace jams {

class BoseEinsteinCudaSRK4Thermostat : public Thermostat {
public:
    BoseEinsteinCudaSRK4Thermostat(const double &temperature, const double &sigma, const double timestep, const int num_spins);
    ~BoseEinsteinCudaSRK4Thermostat() override = default;

    void update() override;

private:
    void warmup(const unsigned steps);

    bool is_warmed_up_ = false;
    unsigned num_warm_up_steps_ = 0;
    double delta_tau_;

    jams::MultiArray<double, 1> w5_;
    jams::MultiArray<double, 1> v5_;
    jams::MultiArray<double, 1> w6_;
    jams::MultiArray<double, 1> v6_;
    jams::MultiArray<double, 1> psi5_;
    jams::MultiArray<double, 1> psi6_;

    CudaStream dev_stream5_;
    CudaStream dev_stream6_;
};

}

#endif  // CUDA
#endif  // INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4
