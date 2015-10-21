__global__ void cuda_zeeman_energy_kernel(const int num_spins, const double time, const int * dc_local_field,
  const double * ac_local_field, const double * ac_local_frequency, const double * dev_s, double * dev_e) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    double e_total = 0.0;
    for (int n = 0; n < 3; ++n) {
      e_total += dev_s[3 * idx + n] * (dc_local_field[3 * idx + n]
       + ac_local_field[3 * idx + n] * cos(ac_local_frequency[idx] * time));
    }
    dev_e[idx] = e_total;
  }
}

__global__ void cuda_zeeman_field_kernel(const int num_spins, const double time, const double * dc_local_field,
  const double * ac_local_field, const double * ac_local_frequency, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    for (int n = 0; n < 3; ++n) {
      dev_h[3 * idx + n] = dc_local_field[3 * idx + n]
      + ac_local_field[3 * idx + n] * cos(ac_local_frequency[idx] * time);
    }
  }
}