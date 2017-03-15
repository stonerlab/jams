__global__ void cuda_zeeman_energy_kernel(const unsigned int num_spins, const double time, const unsigned int * dc_local_field,
  const double * ac_local_field, const double * ac_local_frequency, const double * dev_s, double * dev_e) {

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

      double e_total = 0.0;
    if (idx < num_spins) {
      for (unsigned int n = 0; n < 3; ++n) {
        e_total += dev_s[3 * idx + n] * (dc_local_field[3 * idx + n]
         + ac_local_field[3 * idx + n] * cos(ac_local_frequency[idx] * time));
      }
      dev_e[idx] = e_total;
    }
  }

__global__ void cuda_zeeman_ac_field_kernel(const unsigned int num_spins, const double time,
  const double * ac_local_field, const double * ac_local_frequency, const double * dev_s, double * dev_h) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int gxy = 3 * idx + idy;

  if (idx < num_spins && idy < 3) {
    dev_h[gxy] += ac_local_field[gxy] * cos(ac_local_frequency[idx] * time);
  }
}