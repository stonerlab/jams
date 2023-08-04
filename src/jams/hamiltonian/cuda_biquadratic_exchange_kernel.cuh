// cuda_biquadratic_exchange_kernel.cuh                                -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_BIQUADRATIC_EXCHANGE_KERNEL
#define INCLUDED_JAMS_CUDA_BIQUADRATIC_EXCHANGE_KERNEL

__global__ void cuda_biquadratic_exchange_field_kernel(
    const unsigned int num_spins, const double * dev_s, const int * dev_rows, const int * dev_cols, const double * dev_vals, double * dev_h) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_spins) {
    double h_i[3] = {0.0, 0.0, 0.0};

    double s_i[3] = {dev_s[3*idx + 0], dev_s[3*idx + 1], dev_s[3*idx + 2]};

    for (auto m = dev_rows[idx]; m < dev_rows[idx + 1]; ++m) {
      auto j = dev_cols[m];
      double s_j[3] = {dev_s[3*j + 0], dev_s[3*j + 1], dev_s[3*j + 2]};
      double B_ij = dev_vals[m];

      double s_i_dot_s_j = s_i[0] * s_j[0] + s_i[1] * s_j[1] + s_i[2] * s_j[2];

      for (auto n = 0; n < 3; ++n) {
        h_i[n] += 2.0 * B_ij * s_j[n] * s_i_dot_s_j;
      }
    }

    for (auto n = 0; n < 3; ++n) {
      dev_h[3*idx + n] = h_i[n];
    }
  }
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------