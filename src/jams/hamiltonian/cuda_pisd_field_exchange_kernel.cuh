// cuda_pisd_field_exchange_kernel.cuh                                -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_PISD_FIELD_EXCHANGE_KERNEL
#define INCLUDED_JAMS_CUDA_PISD_FIELD_EXCHANGE_KERNEL

#include <jams/helpers/consts.h>

__global__ void cuda_pisd_field_exchange_field_kernel(
    const unsigned int num_spins, const double Bx, const double By, const double Bz, const double beta, const double * dev_mus, const double * dev_s, const int * dev_rows, const int * dev_cols, const double * dev_vals, double * dev_h) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_spins) {

    const double gmuB = kElectronGFactor * kBohrMagnetonIU;
    const double S = dev_mus[idx] / gmuB;

    double nx1 = dev_s[3*idx + 0];
    double ny1 = dev_s[3*idx + 1];
    double nz1 = dev_s[3*idx + 2];


    double h_i[3] = {0, 0, 0};

    h_i[0] -= 0.5*beta*gmuB*(Bx*(By*ny1 + Bz*nz1) - nx1*(By*By + Bz*Bz));
    h_i[1] -= 0.5*beta*gmuB*(By*(Bx*ny1 + Bz*nz1) - ny1*(By*By + Bz*Bz));
    h_i[2] -= 0.5*beta*gmuB*(Bz*(Bx*ny1 + By*ny1) - nz1*(By*By + Bz*Bz));


    for (auto m = dev_rows[idx]; m < dev_rows[idx + 1]; ++m) {
      auto j = dev_cols[m];
      double nx2 = dev_s[3*j + 0];
      double ny2 = dev_s[3*j + 1];
      double nz2 = dev_s[3*j + 2];

      double J = dev_vals[m];

      h_i[0] -= (0.5*beta*J*J*S*S/gmuB)*(nx2*(ny1*ny2+nz1*nz2));
      h_i[1] -= (0.5*beta*J*J*S*S/gmuB)*(ny2*(nx1*nx2+nz1*nz2));
      h_i[2] -= (0.5*beta*J*J*S*S/gmuB)*(nz2*(nx1*nx2+ny1*ny2));

      h_i[0] += (0.5*beta*J*J*S/gmuB)*(nx1*nx2*nx2);
      h_i[1] += (0.5*beta*J*J*S/gmuB)*(ny1*ny2*ny2);
      h_i[2] += (0.5*beta*J*J*S/gmuB)*(nz1*nz2*nz2);

      h_i[0] += (0.25*beta*J*J*S*(4*S+1)/gmuB)*(nx1*(ny2*ny2 + nz2*nz2));
      h_i[1] += (0.25*beta*J*J*S*(4*S+1)/gmuB)*(ny1*(nx2*nx2 + nz2*nz2));
      h_i[2] += (0.25*beta*J*J*S*(4*S+1)/gmuB)*(nz1*(nx2*nx2 + ny2*ny2));

      h_i[0] -= (0.5*beta*J*S)*(Bx*(ny2*(ny1-ny2) + nz2*(nz1-nz2)));
      h_i[1] -= (0.5*beta*J*S)*(By*(nx2*(nx1-nx2) + nz2*(nz1-nz2)));
      h_i[2] -= (0.5*beta*J*S)*(Bz*(nx2*(nx1-nx2) + ny2*(ny1-ny2)));

      h_i[0] -= (0.5*beta*J*S)*(By*(nx2*(ny1+ny2) - 2*nx1*ny2) + Bz*(nx2*(nz1 + nz2) - 2*nx1*nz2));
      h_i[1] -= (0.5*beta*J*S)*(Bx*(ny2*(nx1+nx2) - 2*ny1*nx2) + Bz*(ny2*(nz1 + nz2) - 2*ny1*nz2));
      h_i[2] -= (0.5*beta*J*S)*(Bx*(nz2*(nx1+nx2) - 2*nz1*nx2) + By*(nz2*(ny1 + ny2) - 2*nz1*ny2));
    }

    for (auto n = 0; n < 3; ++n) {
      dev_h[3*idx + n] = h_i[n];
    }
  }
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------