// cuda_device_rk4.cuh                                                 -*-C++-*-
#ifndef INCLUDED_JAMS_NOISE_CUDA_DEVICE_RK4
#define INCLUDED_JAMS_NOISE_CUDA_DEVICE_RK4

#include <nvfunctional>

template<unsigned N, typename... Args>
__device__ void
rk4(nvstd::function<void(double[N], const double[N], Args...)> const& ode_rhs, double y[N], double h, Args... args) {

  double k1[N], k2[N], k3[N], k4[N], u[N];

  for (auto n = 0; n < N; ++n) {
    u[n] = y[n];
  }

  ode_rhs(k1, u, args...);

  for (auto n = 0; n < N; ++n) {
    u[n] = y[n] + 0.5 * h * k1[n];
  }

  ode_rhs(k2, u, args...);

  for (auto n = 0; n < N; ++n) {
    u[n] = y[n] + 0.5 * h * k2[n];
  }

  ode_rhs(k3, u, args...);

  for (auto n = 0; n < N; ++n) {
    u[n] = y[n] + h * k3[n];
  }

  ode_rhs(k4, u, args...);

  for (auto n = 0; n < N; ++n) {
    y[n] = y[n] + h * (k1[n] + 2*k2[n] + 2*k3[n] + k4[n]) / 6.0;
  }
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------