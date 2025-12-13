__global__ void cuda_rk4_mid_step_kernel
(
  const unsigned dev_num_spins3,
  const double step,
  const double * s_old_dev,
  const double * k_dev,
  double * s_dev

  ) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= dev_num_spins3) return;

  s_dev[idx] = s_old_dev[idx] + step * k_dev[idx];
}
__global__ void cuda_rk4_combination_kernel
    (
        double * s_dev,
        const double * s_old,
        const double * k1_dev,
        const double * k2_dev,
        const double * k3_dev,
        const double * k4_dev,
        const double dt,
        const unsigned dev_num_spins3
    )
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_num_spins3) {
    s_dev[idx] = s_old[idx] + dt * (k1_dev[idx] + 2*k2_dev[idx] + 2*k3_dev[idx] + k4_dev[idx]) / 6.0;
  }
}

__global__ void cuda_spin_normalisation_kernel
    (
        double * s_dev,
        const double * s_old,
        const double * k1_dev,
        const double * k2_dev,
        const double * k3_dev,
        const double * k4_dev,
        const double dt,
        const unsigned dev_num_spins
    )
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_num_spins) {

    double s[3];
    for (auto n = 0; n < 3; ++n) {
      s[n] = s_old[3*idx + n] + dt * (k1_dev[3*idx + n] + 2*k2_dev[3*idx + n] + 2*k3_dev[3*idx + n] + k4_dev[3*idx + n]) / 6.0;
    }

    double recip_snorm = rsqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);

    for (auto n = 0; n < 3; ++n) {
      s_dev[3*idx + n] = s[n] * recip_snorm;
    }
  }
}