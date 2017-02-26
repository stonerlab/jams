__global__ void cuda_anisotropy_cubic_energy_kernel(const int num_spins, 
  const double * mca_value, const double * dev_s, double * dev_e) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int idx3 = 3*idx;

  if (idx < num_spins) {
    const double cubic = dev_s[idx3 + 0] * dev_s[idx3 + 1] * dev_s[idx3 + 2];

    dev_e[idx] = mca_value[idx] * (cubic * cubic);
  }
}

__global__ void cuda_anisotropy_cubic_field_kernel(const int num_spins, 
  const double * mca_value, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int idx3 = 3*idx;
  if (idx < num_spins) {
    const double sx = dev_s[idx3 + 0];
    const double sy = dev_s[idx3 + 1];
    const double sz = dev_s[idx3 + 2];

    dev_h[idx3 + 0] = -2 * mca_value[idx] * sx * sy * sy * sz * sz;
    dev_h[idx3 + 1] = -2 * mca_value[idx] * sx * sx * sy * sz * sz;
    dev_h[idx3 + 2] = -2 * mca_value[idx] * sx * sx * sy * sy * sz;
  }
}