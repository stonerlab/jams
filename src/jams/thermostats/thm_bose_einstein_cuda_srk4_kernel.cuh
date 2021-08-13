// thm_bose_einstein_cuda_srk4_kernel.cuh                              -*-C++-*-

#ifndef INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4_KERNEL
#define INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4_KERNEL

// Implements as **stochastic** fourth order Runge-Kutta scheme (SRK4) to solve
// the second order stochastic differential equation
//
// d^2 X / dt^2 = eta(t) - omega^2 X(t) - gamma dX/dt
//
// where eta(t) is a stochastic term which is a white noise with the correlation
//
// <eta(t), eta(0)> = 2 gamma delta(t)
//
// In code we solve the differential equation as a set of coupled first order
// differential equations
//
// dV/dt = W(t)
//
// dW/dt = eta - omega^2 V(t) - gamma W(t)
//
// The SRK4 is derived in Honeycutt, Phys. Rev. A 45, 604 (1992)
// https://dx.doi.org/10.1103/physreva.45.604
//
// Usually stochastic differential equations are solved with a fourth order
// Runge Kutta where the stochastic part is held constant across the step.
// However this is not a correct scheme (the order of the scheme for the
// stochastic part is lower than that of the deterministic part). Using a
// proper fourth order stochastic integration scheme requires two gaussian
// random numbers per step but should ensure better adherence to the required
// correlation properties at larger step sizes.
//
//
// The stochastic part of the equation can be reduced from four to two
// white noise processess, but constants are required to combine these
// together at each sub-step of the whole Runge-Kutta step. The constants are
// given in p607, Honeycutt, Phys. Rev. A 45, 604 (1992)
//
// a1 = 1/4 + sqrt(3)/6        b1 = 1/4 - sqrt(3)/6 + sqrt(6)/12
// a2 = 1/4 + sqrt(3)/6        b2 = 1/4 - sqrt(3)/6 - sqrt(6)/12
// a3 = 1/2 + sqrt(3)/6        b3 = 1/2 - sqrt(3)/6
// a4 = 5/4 + sqrt(3)/6        b4 = 5/4 - sqrt(3)/6 + sqrt(6)/12
//
// I've generated these to maximum double precision using the following code:
//----------------------------------------------------------------------------
//  #include <cmath>
//  #include <iomanip>
//  #include <iostream>
//  #include <limits>
//  #include <map>
//  #include <string>
//
//  int main(int argc, char *argv[]) {
//
//    std::map<std::string, std::function<double()>> map =
//      {
//      {"a1", []()->double {return 1.0 / 4.0 + sqrt(3.0) / 6.0; }},
//      {"a2", []()->double {return 1.0 / 4.0 + sqrt(3.0) / 6.0; }},
//      {"a3", []()->double {return 1.0 / 2.0 + sqrt(3.0) / 6.0; }},
//      {"a4", []()->double {return 5.0 / 4.0 + sqrt(3.0) / 6.0; }},
//      {"b1", []()->double {return 1.0 / 4.0 - sqrt(3.0) / 6.0 + sqrt(6.0) / 12.0; }},
//      {"b2", []()->double {return 1.0 / 4.0 - sqrt(3.0) / 6.0 - sqrt(6.0) / 12.0; }},
//      {"b3", []()->double {return 1.0 / 2.0 - sqrt(3.0) / 6.0; }},
//      {"b4", []()->double {return 5.0 / 4.0 - sqrt(3.0) / 6.0 + sqrt(6.0) / 12.0; }},
//      };
//
//    std::cout.precision(std::numeric_limits<double>::max_digits10 + 1);
//    for (auto const& x : map) {
//      std::cout << "const double " << x.first << " = " << x.second() << ";" << std::endl;
//    }
//  }
//----------------------------------------------------------------------------
//

namespace jams {

/// @brief This kernel performs one step of the SRK4 algorithm for the coupled
/// stochastic differential equations:
///
///     dV/dt = W(t);     dW/dt = eta - omega^2 V(t) - gamma W(t)
///
/// where eta(t) is a stochastic term which is a white noise with the correlation
///
///     <eta(t), eta(0)> = 2 gamma delta(t)
///
/// @param w_dev     device pointer to data for W(t)
/// @param v_dev     device pointer to data for V(t)
/// @param psi_dev   device pointer to data for psi(t) (2*size standard gaussian noise processes)
/// @param omega     omega constant in the differential equation
/// @param gamma     gamma constant in the differential equation
/// @param h         integration step size
/// @param size      number of processes to solve simultaneously (length of device arrays)
__global__ void stochastic_rk4_cuda_kernel(
    double * w_dev,
    double * v_dev,
    const double * psi_dev,
    const double omega,
    const double gamma,
    const double h, // step size
    const unsigned size) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // From p607, Honeycutt, Phys. Rev. A 45, 604 (1992) (see documentation at
  // the top of this file).
  const double a1 = 0.538675134594812866;
  const double a2 = 0.538675134594812866;
  const double a3 = 0.788675134594812866;
  const double a4 = 1.53867513459481287;
  const double b1 = 0.165449010637118615;
  const double b2 = -0.242799279826744346;
  const double b3 = 0.211324865405187134;
  const double b4 = 1.1654490106371187;

  // The expression inside the sqrt() is half of what we would normally have
  // because the integration scheme uses two random gaussian samples.
  // Look through and compare Eqs. (1.5), (2.4b), (3.9b), (4.4b), (4.10b) in
  // Honeycutt, Phys. Rev. A 45, 604 (1992)
  const double sigma = sqrt(gamma * h);

  // If treated with a normal (not-stochastic) fourth order Runge-Kutta this
  // function would return
  //
  //    eta - omega * omega * v - gamma * w;
  //
  // where eta would be the stochastic part. However, in the SRK4 scheme the
  // stochastic part is not included in the right-hand
  // side of the equation explicitly but comes in through the integration
  // scheme (see Eqs. (1.3b) and (4.11a) in Honeycutt, Phys. Rev. A 45, 604
  // (1992), where their 'epsilon' is our 'w'. The function 'h(epsilon)' is
  // a the non-stochastic part of the RHS of the differential equation but
  // within the integration scheme expressions like
  //
  //    h(epsilon_0 + sigma * (a1 * psi1 + b1 * psi2))
  //
  // appear, meaning the noise is being added to the variable epsilon (w)
  // directly.
  auto w_rhs = [&](const double& w, const double& v) {
    return - omega * omega * v - gamma * w;
  };

  auto v_rhs = [](const double& w) {
    return w;
  };

  if (idx < size) {

    const double w0 = w_dev[idx];
    const double v0 = v_dev[idx];
    const double psi1 = psi_dev[2*idx + 0];
    const double psi2 = psi_dev[2*idx + 1];

    const double w_k1 = w_rhs(
        w0 + sigma * (a1 * psi1 + b1 * psi2),
        v0);
    const double v_k1 = v_rhs(
        w0 + sigma * (a1 * psi1 + b1 * psi2));

    const double w_k2 = w_rhs(
        w0 + 0.5 * h * w_k1 + sigma * (a2 * psi1 + b2 * psi2),
        v0 + 0.5 * h * v_k1);
    const double v_k2 = v_rhs(
        w0 + 0.5 * h * w_k1 + sigma * (a2 * psi1 + b2 * psi2));

    const double w_k3 = w_rhs(
        w0 + 0.5 * h * w_k2 + sigma * (a3 * psi1 + b3 * psi2),
        v0 + 0.5 * h * v_k2);
    const double v_k3 = v_rhs(
        w0 + 0.5 * h * w_k2 + sigma * (a3 * psi1 + b3 * psi2));

    const double w_k4 = w_rhs(
        w0 + h * w_k3 + sigma * (a4 * psi1 + b4 * psi2),
        v0 + h * v_k3);
    const double v_k4 = v_rhs(
        w0 + h * w_k3 + sigma * (a4 * psi1 + b4 * psi2));

    w_dev[idx] = w0 + h * (w_k1 + 2.0 * w_k2 + 2.0 * w_k3 + w_k4) / 6.0 + sigma * (psi1 + psi2);
    v_dev[idx] = v0 + h * (v_k1 + 2.0 * v_k2 + 2.0 * v_k3 + v_k4) / 6.0;
  }
}

/// This kernel combines two stochastic processes into a single noise term
/// which depends on the temperature and the local parameters on each spin
/// site.
__global__ void stochastic_combination_cuda_kernel(
    double * noise,
    const double * v5_dev,
    const double * v6_dev,
    const double *sigma,
    const double T,
    const unsigned size) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // The two magic numbers here are "c_5" and "c_6" in Table 1, Savin,
    // Phys. Rev. B 86, 064305 (2012) https://dx.doi.org/10.1103/physrevb.86.064305
    noise[idx] = T * sigma[idx] * (1.8315 * v5_dev[idx] + 0.3429 * v6_dev[idx]);
  }
}

}

#endif  // INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4_KERNEL
