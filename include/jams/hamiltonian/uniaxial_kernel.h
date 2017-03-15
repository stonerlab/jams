// uniaxial anisotropy is an expansion in legendre polynomials so lets define upto n = 6
// so we have some fast intrinsics

__device__ inline double cuda_legendre_poly_0(const double x) {
  return 1.0;
}

__device__ inline double cuda_legendre_poly_1(const double x) {
  return x;
}

__device__ inline double cuda_legendre_poly_2(const double x) {
  // (3x^2 - 1)/2
  return (1.5 * x * x - 0.5);
}

__device__ inline double cuda_legendre_poly_3(const double x) {
  // (5x^3 - 3x)/2
  return (2.5 * x * x * x - 1.5 * x);
}

__device__ inline double cuda_legendre_poly_4(const double x) {
  // (35x^4 - 30x^2 + 3)/8
  return (4.375 * x * x * x * x - 3.75 * x * x + 0.375);
}

__device__ inline double cuda_legendre_poly_5(const double x) {
  // (63x^5 - 70x^3 + 15x)/8
  return (7.875 * x * x * x * x * x - 8.75 * x * x * x + 1.875 * x);
}

__device__ inline double cuda_legendre_poly_6(const double x) {
  // (231x^6 - 315x^4 + 105x^2 - 5)/16
  return (14.4375 * x * x * x * x * x * x - 19.6875 * x * x * x * x + 6.5625 * x * x - 0.3125);
}

__device__ inline double cuda_legendre_dpoly_0(const double x) {
  return 0.0;
}

__device__ inline double cuda_legendre_dpoly_1(const double x) {
  return 1.0;
}

__device__ inline double cuda_legendre_dpoly_2(const double x) {
  return 3.0 * x;
}

__device__ inline double cuda_legendre_dpoly_3(const double x) {
  return (7.5 * x * x - 1.5);
}

__device__ inline double cuda_legendre_dpoly_4(const double x) {
  return (17.5 * x * x * x - 7.5 * x);
}

__device__ inline double cuda_legendre_dpoly_5(const double x) {
  return (39.375 * x * x * x * x - 26.25 * x * x + 1.875);
}

__device__ inline double cuda_legendre_dpoly_6(const double x) {
  return (86.625 * x * x * x * x * x - 78.75 * x * x * x + 13.125 * x);
}


__device__ double cuda_legendre_poly(const double x, const int n) {

  switch (n)
  {
    case 0:
      return cuda_legendre_poly_0(x);
    case 1:
      return cuda_legendre_poly_1(x);
    case 2:
      return cuda_legendre_poly_2(x);
    case 3:
      return cuda_legendre_poly_3(x);
    case 4:
      return cuda_legendre_poly_4(x);
    case 5:
      return cuda_legendre_poly_5(x);
    case 6:
      return cuda_legendre_poly_6(x);
  }

  // http://www.storage-b.com/math-numerical-analysis/18
  double pn1(cuda_legendre_poly_2(x));
  double pn2(cuda_legendre_poly_1(x));
  double pn(pn1);

  for (int l = 3; l < n + 1; ++l) {
    pn =  (((2.0 * l) - 1.0) * x * pn1 - ((l - 1.0) * pn2)) / static_cast<double>(l);
    pn2 = pn1;
    pn1 = pn;
  }

  return pn;
}

__device__ double cuda_legendre_dpoly(const double x, const int n) {

  switch (n)
  {
    case 0:
      return cuda_legendre_dpoly_0(x);
    case 1:
      return cuda_legendre_dpoly_1(x);
    case 2:
      return cuda_legendre_dpoly_2(x);
    case 3:
      return cuda_legendre_dpoly_3(x);
    case 4:
      return cuda_legendre_dpoly_4(x);
    case 5:
      return cuda_legendre_dpoly_5(x);
    case 6:
      return cuda_legendre_dpoly_6(x);
  }

  return (x * cuda_legendre_poly(x, n) - cuda_legendre_poly(x, n - 1)) / static_cast<double>(2 * n + 1);
}


__global__ void cuda_uniaxial_energy_kernel(const int num_spins, const int num_mca, const int * mca_order,
  const double * mca_value, const double * dev_s, double * dev_e) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double sz = dev_s[3*idx+2];
    double energy = 0.0;
    for (int n = 0; n < num_mca; ++n) {
      energy += mca_value[num_mca * idx + n] * cuda_legendre_poly(sz, mca_order[n]);
    }
    dev_e[idx] =  energy;
  }
}

__global__ void cuda_uniaxial_field_kernel(const int num_spins, const int num_mca, const int * mca_order,
  const double * mca_value, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double sz = dev_s[3*idx+2];
    double hz = 0.0;
    for (int n = 0; n < num_mca; ++n) {
      hz += -mca_value[num_mca * idx + n] * cuda_legendre_dpoly(sz, mca_order[n]);
    }
    dev_h[3 * idx + 2] =  hz;
  }
}

// __global__ void cuda_uniaxial_field_kernel(const int num_spins, const double * dev_d2z, const double * dev_d4z,
//   const double * dev_d6z, const double * dev_s, double * dev_h) {
//   const int idx = blockIdx.x*blockDim.x+threadIdx.x;
//   if (idx < num_spins) {
//     const double sz = dev_s[3*idx+2];
//     dev_h[3*idx+2] = -dev_d2z[idx]*3.0*sz - dev_d4z[idx]*(17.5*sz*sz*sz-7.5*sz) - dev_d6z[idx]*(86.625*sz*sz*sz*sz*sz - 78.75*sz*sz*sz + 13.125*sz);
//   }
// }

// __global__ void cuda_uniaxial_energy_kernel(const int num_spins, const double * dev_d2z, const double * dev_d4z,
//   const double * dev_d6z, const double * dev_s, double * dev_e) {
//   const int idx = blockIdx.x*blockDim.x+threadIdx.x;
//   if (idx < num_spins) {
//     const double sz = dev_s[3*idx+2];
//     dev_e[idx] =  0.5*dev_d2z[idx]*(3.0*sz*sz - 1.0) + 0.125*dev_d4z[idx]*(35.0*sz*sz*sz*sz - 30.0*sz*sz + 3.0)
//                       + 0.0625*dev_d6z[idx]*(231.0*sz*sz*sz*sz*sz*sz + 315.0*sz*sz*sz*sz + 105.0*sz*sz - 5.0);
//   }
// }