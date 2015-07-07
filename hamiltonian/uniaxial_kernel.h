__global__ void cuda_uniaxial_field_kernel(const int num_spins, const double * dev_d2z, const double * dev_d4z,
  const double * dev_d6z, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double sz = dev_s[3*idx+2];
    dev_h[3*idx+2] += dev_d2z[idx]*3.0*sz + dev_d4z[idx]*(17.5*sz*sz*sz-7.5*sz) + dev_d6z[idx]*(86.625*sz*sz*sz*sz*sz - 78.75*sz*sz*sz + 13.125*sz);
  }
}

__global__ void cuda_uniaxial_energy_kernel(const int num_spins, const double * dev_d2z, const double * dev_d4z,
  const double * dev_d6z, const double * dev_s, double * dev_e) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double sz = dev_s[3*idx+2];
    dev_e[idx] = dev_e[idx] - 0.5*dev_d2z[idx]*(3.0*sz*sz - 1.0) - 0.125*dev_d4z[idx]*(35.0*sz*sz*sz*sz - 30.0*sz*sz + 3.0)
                      - 0.0625*dev_d6z[idx]*(231.0*sz*sz*sz*sz*sz*sz - 315.0*sz*sz*sz*sz + 105.0*sz*sz - 5.0);
  }
}