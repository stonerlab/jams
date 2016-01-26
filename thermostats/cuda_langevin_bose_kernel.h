// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H


__device__ void linear_ode(const double A[4], const double eta[4], const double z[4], double f[4]) {
    for (int i = 0; i < 4; ++i) {
        f[i] = A[i]*(eta[i] - z[i]);
    }
}


__device__ void bose_ode(const double A[4], const double eta[4], const double z[4], double f[4]) {
    for (int i = 0; i < 2; ++i) {
        int n = 2*i;
        f[n] = z[n+1];
        f[n+1] = eta[i] - A[i+2]*A[i+2]*z[n] - A[i]*z[n+1];
    }
}

__device__ void rk4(void ode(const double[4], const double[4], const double[4], double[4]), const double h, const double A[4], const double eta[4], double z[4]) {
    int i;
    double k1[4], k2[4], k3[4], k4[4], f[4];
    double u[4] = {z[0], z[1], z[2], z[3]};

    ode(A, eta, u, f);
    for (i = 0; i < 4; ++i) {
        k1[i] = h*f[i];
    }

    // K2
    for (i = 0; i < 4; ++i) {
        u[i] = z[i] + 0.5 * k1[i];
    }
    ode(A, eta, u, f);
    for (i = 0; i < 4; ++i) {
        k2[i] = h*f[i];
    }

    // K3
    for (i = 0; i < 4; ++i) {
        u[i] = z[i] + 0.5 * k2[i];
    }
    ode(A, eta, u, f);
    for (i = 0; i < 4; ++i) {
        k3[i] = h*f[i];
    }

    // K4
    for (i = 0; i < 4; ++i) {
        u[i] = z[i] + k3[i];
    }
    ode(A, eta, u, f);
    for (i = 0; i < 4; ++i) {
        k4[i] = h*f[i];
    }

    for (i = 0; i < 4; ++i) {
        z[i] = z[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
}

__global__ void coth_stochastic_process_cuda_kernel
(
          double * noise,
          double * zeta,
    const double * eta,
    const double h,
    const double T,
    const double w_m,
    const int    N
) {
    const double c[6] = {1.043576*w_m,
                         0.177222*w_m,
                         0.050319*w_m,
                         0.010241*w_m,
                         1.8315,
                         0.3429};

    double s0 = 0.0, s1 = 0.0;

    int i;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < N) {
        // ------------------------------------------------------
        const double lambda[4] = {1.763817*w_m,
                                  0.394613*w_m,
                                  0.103506*w_m,
                                  0.015873*w_m};

        double z[4] = {zeta[8*x],
                       zeta[8*x+1],
                       zeta[8*x+2],
                       zeta[8*x+3]};

        double e[4] = {eta[6*x]*sqrt(2.0/lambda[0]),
                       eta[6*x+1]*sqrt(2.0/lambda[1]),
                       eta[6*x+2]*sqrt(2.0/lambda[2]),
                       eta[6*x+3]*sqrt(2.0/lambda[3])};

        rk4(linear_ode, h*w_m, lambda, e, z);

        for (i = 0; i < 4; ++i) {
            zeta[8*x+i] = z[i];
        }

        for (i = 0; i < 4; ++i) {
            s0 += c[i]*(e[i] - z[i]);
        }
        // ------------------------------------------------------

        // first two elements are gamma, second two are omega
        const double gamma_omega[4] = {5.0142, 3.2974, 2.7189, 1.2223};
        for (i = 0; i < 4; ++i) {
            z[i] = zeta[8*x+i+4];
        }

        for (i = 0; i < 2; ++i) {
            e[i] = eta[6*x+i+4]*sqrt(2.0*gamma_omega[i]);
        }
        e[2] = 0.0;
        e[3] = 0.0;

        rk4(bose_ode, h, gamma_omega, e, z);

        for (i = 0; i < 4; ++i) {
            zeta[8*x+4+i] = z[i];
        }
        // ------------------------------------------------------

        s1 = c[4]*z[0] + c[5]*z[2];
        noise[x] = T*(s0 + s1);
    }
}

#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H
