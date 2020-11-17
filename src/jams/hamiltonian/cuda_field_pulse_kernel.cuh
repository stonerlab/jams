#ifndef JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_KERNEL_CUH
#define JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_KERNEL_CUH

__global__ void cuda_field_pulse_surface_kernel(const unsigned int num_spins, const double surface_cutoff, const double * dev_mus, const double * dev_r, const double3 b_field, double * dev_h) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_spins) {
    // check z component of spin position
    if (dev_r[3*idx + 2] > surface_cutoff) {
      dev_h[3*idx + 0] = dev_mus[idx] * b_field.x;
      dev_h[3*idx + 1] = dev_mus[idx] * b_field.y;
      dev_h[3*idx + 2] = dev_mus[idx] * b_field.z;
    } else {
      dev_h[3*idx + 0] = 0.0;
      dev_h[3*idx + 1] = 0.0;
      dev_h[3*idx + 2] = 0.0;    }
  }
}

#endif //JAMS_HAMILTONIAN_CUDA_FIELD_PULSE_KERNEL_CUH
