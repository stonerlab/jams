#include <cuda.h>
#include <cufft.h>

__global__ void spectrum(
    const int i,
    const int j,
    const int num_spins,
    const int num_k,
    const int num_freq,
    const double qx,
    const double qy,
    const double qz,
    const double rij_x,
    const double rij_y,
    const double rij_z,
    const double* kvectors,
    const cufftDoubleComplex* spin_frequencies,
    cufftDoubleComplex* skw)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int w = blockIdx.y * blockDim.y + threadIdx.y;

  if (k < num_k && w < num_freq) {


    const double unit_q[3] = {qx, qy, qz};

    cufftDoubleComplex s_iw[3] = {
        spin_frequencies[3 * (w * num_spins  + i) + 0],
        spin_frequencies[3 * (w * num_spins  + i) + 1],
        spin_frequencies[3 * (w * num_spins  + i) + 2]};
    cufftDoubleComplex s_jw[3] = {
        spin_frequencies[3 * (w * num_spins  + j) + 0],
        spin_frequencies[3 * (w * num_spins  + j) + 1],
        spin_frequencies[3 * (w * num_spins  + j) + 2]};

    cufftDoubleComplex s_ij_w = {0.0, 0.0};
    for (auto m = 0; m < 3; ++m) {
      for (auto n = 0; n < 3; ++n) {
        const auto geom = (double(m == n) - unit_q[m] * unit_q[n]);
        if (geom == 0.0) continue;
        // conj(s_imw) * s_jnw
        s_ij_w.x += geom * (s_iw[m].x * s_jw[n].x + s_iw[m].y * s_jw[n].y);
        s_ij_w.y += geom * (s_iw[m].x * s_jw[n].y - s_iw[m].y * s_jw[n].x);
      }
    }

    const double3 kvec = {kvectors[3*k + 0], kvectors[3*k + 1], kvectors[3*k + 2]};

    double k_dot_r = kvec.x * rij_x + kvec.y * rij_y + kvec.z * rij_z;
    cufftDoubleComplex exp_kr = {cos(-2*M_PI*k_dot_r), sin(-2*M_PI*k_dot_r)};

    skw[k * num_freq + w].x += (exp_kr.x*s_ij_w.x - exp_kr.y*s_ij_w.y);
    skw[k * num_freq + w].y += (exp_kr.x*s_ij_w.y + exp_kr.y*s_ij_w.x);
  }
}

__global__ void spectrum_i_equal_j(
    const int i,
    const int j,
    const int num_spins,
    const int num_k,
    const int num_freq,
    const double qx,
    const double qy,
    const double qz,
    const cufftDoubleComplex* spin_frequencies,
    cufftDoubleComplex* skw)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int w = blockIdx.y * blockDim.y + threadIdx.y;

  if (k < num_k && w < num_freq) {


    const double unit_q[3] = {qx, qy, qz};

    cufftDoubleComplex s_iw[3] = {
        spin_frequencies[3 * (w * num_spins  + i) + 0],
        spin_frequencies[3 * (w * num_spins  + i) + 1],
        spin_frequencies[3 * (w * num_spins  + i) + 2]};

    cufftDoubleComplex s_ij_w = {0.0, 0.0};
    for (auto m = 0; m < 3; ++m) {
      for (auto n = 0; n < 3; ++n) {
        const auto geom = (double(m == n) - unit_q[m] * unit_q[n]);
        if (geom == 0.0) continue;
        // conj(s_imw) * s_jnw
        s_ij_w.x += geom * (s_iw[m].x * s_iw[n].x + s_iw[m].y * s_iw[n].y);
        s_ij_w.y += geom * (s_iw[m].x * s_iw[n].y - s_iw[m].y * s_iw[n].x);
      }
    }

    skw[k * num_freq + w].x += s_ij_w.x;
    skw[k * num_freq + w].y += s_ij_w.y;
  }
}

__global__ void spectrum_i_not_equal_j(
    const int i,
    const int j,
    const int num_spins,
    const int num_k,
    const int num_freq,
    const double qx,
    const double qy,
    const double qz,
    const double rij_x,
    const double rij_y,
    const double rij_z,
    const double* kvectors,
    const cufftDoubleComplex* spin_frequencies,
    cufftDoubleComplex* skw)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int w = blockIdx.y * blockDim.y + threadIdx.y;

  if (k < num_k && w < num_freq) {


    const double unit_q[3] = {qx, qy, qz};

    cufftDoubleComplex s_iw[3] = {
        spin_frequencies[3 * (w * num_spins  + i) + 0],
        spin_frequencies[3 * (w * num_spins  + i) + 1],
        spin_frequencies[3 * (w * num_spins  + i) + 2]};
    cufftDoubleComplex s_jw[3] = {
        spin_frequencies[3 * (w * num_spins  + j) + 0],
        spin_frequencies[3 * (w * num_spins  + j) + 1],
        spin_frequencies[3 * (w * num_spins  + j) + 2]};

    cufftDoubleComplex s_ij_w = {0.0, 0.0};
    cufftDoubleComplex s_ji_w = {0.0, 0.0};

    for (auto m = 0; m < 3; ++m) {
      for (auto n = 0; n < 3; ++n) {
        const auto geom = (double(m == n) - unit_q[m] * unit_q[n]);
        if (geom == 0.0) continue;
        // conj(s_imw) * s_jnw
        s_ij_w.x += geom * (s_iw[m].x * s_jw[n].x + s_iw[m].y * s_jw[n].y);
        s_ij_w.y += geom * (s_iw[m].x * s_jw[n].y - s_iw[m].y * s_jw[n].x);

        s_ji_w.x += geom * (s_jw[m].x * s_iw[n].x + s_jw[m].y * s_iw[n].y);
        s_ji_w.y += geom * (s_jw[m].x * s_iw[n].y - s_jw[m].y * s_iw[n].x);
      }
    }

    const double3 kvec = {kvectors[3*k + 0], kvectors[3*k + 1], kvectors[3*k + 2]};

    double k_dot_r = kvec.x * rij_x + kvec.y * rij_y + kvec.z * rij_z;
    cufftDoubleComplex exp_kr = {cos(-2*M_PI*k_dot_r), sin(-2*M_PI*k_dot_r)};

    skw[k * num_freq + w].x +=
        (exp_kr.x*s_ij_w.x - exp_kr.y*s_ij_w.y) + (exp_kr.x*s_ji_w.x + exp_kr.y*s_ji_w.y);

    skw[k * num_freq + w].y +=
        (exp_kr.x*s_ij_w.y + exp_kr.y*s_ij_w.x) + (exp_kr.x*s_ji_w.y - exp_kr.y*s_ji_w.x);
  }
}

__global__ void spectrum_r_ij(
    const int i,
    const int num_spins,
    const int num_k,
    const int num_freq,
    const double qx,
    const double qy,
    const double qz,
    const double* r_ij_dev,
    const double* kvectors,
    const cufftDoubleComplex* spin_frequencies,
    cufftDoubleComplex* skw)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int w = blockIdx.y * blockDim.y + threadIdx.y;

  if (k < num_k && w < num_freq) {
    const double unit_q[3] = {qx, qy, qz};
    const double3 kvec = {kvectors[3 * k + 0], kvectors[3 * k + 1],
                          kvectors[3 * k + 2]};

    cufftDoubleComplex s_iw[3] = {
        spin_frequencies[3 * (w * num_spins  + i) + 0],
        spin_frequencies[3 * (w * num_spins  + i) + 1],
        spin_frequencies[3 * (w * num_spins  + i) + 2]};

    cufftDoubleComplex skw_sum = {0.0, 0.0};
    for (auto m = 0; m < 3; ++m) {
      for (auto n = 0; n < 3; ++n) {
        const auto geom = (double(m == n) - unit_q[m] * unit_q[n]);
        if (geom == 0.0) continue;
        // conj(s_imw) * s_jnw
        skw_sum.x += geom * (s_iw[m].x * s_iw[n].x + s_iw[m].y * s_iw[n].y);
        skw_sum.y += geom * (s_iw[m].x * s_iw[n].y - s_iw[m].y * s_iw[n].x);
      }
    }

    for (auto j = i + 1; j < num_spins; ++j) {
      cufftDoubleComplex s_jw[3] = {
          spin_frequencies[3 * (w * num_spins + j) + 0],
          spin_frequencies[3 * (w * num_spins + j) + 1],
          spin_frequencies[3 * (w * num_spins + j) + 2]};


      const double r_ij[3] = {
          r_ij_dev[3*j + 0],
          r_ij_dev[3*j + 1],
          r_ij_dev[3*j + 2]};

      cufftDoubleComplex s_ij_w = {0.0, 0.0};
      cufftDoubleComplex s_ji_w = {0.0, 0.0};

      for (auto m = 0; m < 3; ++m) {
        for (auto n = 0; n < 3; ++n) {
          const auto geom = (double(m == n) - unit_q[m] * unit_q[n]);
          if (geom == 0.0) continue;
          // conj(s_imw) * s_jnw

          s_ij_w.x += geom * (s_iw[m].x * s_jw[n].x + s_iw[m].y * s_jw[n].y);
          s_ij_w.y += geom * (s_iw[m].x * s_jw[n].y - s_iw[m].y * s_jw[n].x);

          s_ji_w.x += geom * (s_jw[m].x * s_iw[n].x + s_jw[m].y * s_iw[n].y);
          s_ji_w.y += geom * (s_jw[m].x * s_iw[n].y - s_jw[m].y * s_iw[n].x);
        }
      }

      double k_dot_r = kvec.x * r_ij[0] + kvec.y * r_ij[1] + kvec.z * r_ij[2];
      cufftDoubleComplex exp_kr = {cos(-2 * M_PI * k_dot_r),
                                   sin(-2 * M_PI * k_dot_r)};

      skw_sum.x +=
          (exp_kr.x*s_ij_w.x - exp_kr.y*s_ij_w.y) + (exp_kr.x*s_ji_w.x + exp_kr.y*s_ji_w.y);

      skw_sum.y +=
          (exp_kr.x*s_ij_w.y + exp_kr.y*s_ij_w.x) + (exp_kr.x*s_ji_w.y - exp_kr.y*s_ji_w.x);
    }

    skw[k * num_freq + w].x += skw_sum.x;
    skw[k * num_freq + w].y += skw_sum.y;

  }
}