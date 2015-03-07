// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_KERNEL_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_KERNEL_H


__device__ void coth_ode(const double eta[6], const double z[8], double f[8]) {

    const double lambda[4] = {1.763817, 0.394613, 0.103506, 0.015873};
    const double omega[2] = {2.7189, 1.2223};
    const double gamma[2] = {5.0142, 3.2974};
    int i;

    for (i = 0; i < 4; ++i) {
        f[i] = lambda[i]*(eta[i] - z[i]);
    }

    for (i = 0; i < 2; ++i) {
        int n = 2*i + 4;
        f[n] = z[n+1];
        f[n+1] = eta[i+4] - omega[i]*omega[i]*z[n] - gamma[i]*z[n+1];
    }
}

__global__ void coth_stochastic_process_cuda_kernel
(
          double * noise,
          double * zeta,
    const double * eta,
    const double h
) {
    const double c[6] = {1.043576, 0.177222, 0.050319, 0.010241, 1.8315, 0.3429};

    int i;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    double z[8] = {zeta[8*x], zeta[8*x+1], zeta[8*x+2], zeta[8*x+3], zeta[8*x+4], zeta[8*x+5], zeta[8*x+6], zeta[8*x+7]};
    double e[6] = {eta[6*x], eta[6*x+1], eta[6*x+2], eta[6*x+3], eta[6*x+4], eta[6*x+5]};

    double k1[8], k2[8], k3[8], k4[8], f[8], u[8];

    // K1
    coth_ode(e, u, f);
    for (i = 0; i < 8; ++i) {
        k1[i] = h*f[i];
    }

    // K2
    for (i = 0; i < 8; ++i) {
        u[i] = z[i] + 0.5 * k1[i];
    }
    coth_ode(e, u, f);
    for (i = 0; i < 8; ++i) {
        k2[i] = h*f[i];
    }

    // K3
    for (i = 0; i < 8; ++i) {
        u[i] = z[i] + 0.5 * k2[i];
    }
    coth_ode(e, u, f);
    for (i = 0; i < 8; ++i) {
        k3[i] = h*f[i];
    }

    // K4
    for (i = 0; i < 8; ++i) {
        u[i] = z[i] + k3[i];
    }
    coth_ode(e, u, f);
    for (i = 0; i < 8; ++i) {
        k4[i] = h*f[i];
    }

    for (i = 0; i < 8; ++i) {
        u[i] = z[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }

    for (i = 0; i < 8; ++i) {
        zeta[8*x+i] = u[i];
    }

    double s0 = 0.0, s1 = 0.0;
    for (i = 0; i < 4; ++i) {
        s0 += c[i]*(e[i] - u[i]);
    }

    s1 = c[4]*u[4] + c[5]*u[6];
    noise[x] = s0 + s1;
}

// This kernel simulates the PSD p(omega) = (1/2)omega
__global__ void linear_stochastic_process_cuda_kernel
(
          double * zeta,
    const double * eta,
    const double h
) {
  const double lambda[4] = {1.763817, 0.394613, 0.103506, 0.015873};

  const int x = blockIdx.x * blockDim.x + threadIdx.x;  // 0...3
  const int y = blockIdx.y * blockDim.y + threadIdx.y;  // 0...3*num_spins-1

  const int grid_width = gridDim.x * blockDim.x;

  const int index = y * grid_width + x;

  double zz = zeta[index];
  double ee = sqrt(2.0/lambda[x])*eta[index];

  double k1 = lambda[x]*(ee - zz);
  double k2 = lambda[x]*(ee - (zz + 0.5*h*k1));
  double k3 = lambda[x]*(ee - (zz + 0.5*h*k2));
  double k4 = lambda[x]*(ee - (zz + h*k3));

  zeta[index] = zz + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
}

__device__ void bose_ode(const double omega, const double gamma, const double eta, const double u[2], double f[2]) {
    f[0] = u[1];
    f[1] = eta - omega*omega*u[0] - gamma*u[1];
}

// This kernel simulates the PSD p(omega) = omega/(exp(omega) - 1)
__global__ void bose_stochastic_process_cuda_kernel
(
          double * zeta,
    const double * eta,
    const double h
) {
    const double omega[2] = {2.7189, 1.2223};
    const double gamma[2] = {5.0142, 3.2974};

    int i;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // 0...1
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // 0...3*num_spins-1

    const int grid_width = gridDim.x * blockDim.x;

    const int index = y * grid_width + x;

    double zz[2] = {zeta[index], zeta[index + 1]};
    double u[2] = {zz[0], zz[1]};
    const double ee = sqrt(2.0*gamma[x])*eta[index];

    double k1[2], k2[2], k3[2], k4[2], f[2];

    // K1
    bose_ode(omega[x], gamma[x], ee, u, f);
    #pragma unroll
    for (i = 0; i < 2; ++i) {
        k1[i] = h*f[i];
    }

    // K2
    #pragma unroll
    for (i = 0; i < 2; ++i) {
        u[i] = zz[i] + 0.5 * k1[i];
    }
    bose_ode(omega[x], gamma[x], ee, u, f);
    #pragma unroll
    for (i = 0; i < 2; ++i) {
        k2[i] = h*f[i];
    }

    // K3
    #pragma unroll
    for (i = 0; i < 2; ++i) {
        u[i] = zz[i] + 0.5 * k2[i];
    }
    bose_ode(omega[x], gamma[x], ee, u, f);
    #pragma unroll
    for (i = 0; i < 2; ++i) {
        k3[i] = h*f[i];
    }

    // K4
    #pragma unroll
    for (i = 0; i < 2; ++i) {
        u[i] = zz[i] + k3[i];
    }
    bose_ode(omega[x], gamma[x], ee, u, f);
    #pragma unroll
    for (i = 0; i < 2; ++i) {
        k4[i] = h*f[i];
    }

    #pragma unroll
    for (i = 0; i < 2; ++i) {
        zeta[index+i] = zz[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
}

__global__ void combine_stochastic_process_cuda_kernel
(
          double * noise,
    const double * zeta_linear,
    const double * zeta_bose,
    const double * eta_linear
) {
    const double c[6] = {1.043576, 0.177222, 0.050319, 0.010241, 1.8315, 0.3429};

    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    double s0 = 0.0, s1 = 0.0;

    // S_{0an}
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
    s0 += c[i]*(eta_linear[4*x + i] - zeta_linear[4*x + i]);
    }

    // S_{1an}
    // index + 2 in the second term because the second order ODE is broken down into two first order ODES in the RK4
    // which are stored as [u1, u2]_1 , [u1 u2]_2 where u1 = zeta in each case
    s1 = c[4]*zeta_bose[4*x] + c[5]*zeta_bose[4*x + 2];

    noise[x] = s0 + s1;
}

#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_COTH_KERNEL_H
