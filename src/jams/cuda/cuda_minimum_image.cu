#include <jams/cuda/cuda_minimum_image.h>
#include <jams/containers/mat3.h>
#include <jams/core/globals.h>

__constant__ jams::Real kMatrix[9];
__constant__ jams::Real kMatrixInv[9];
__constant__ bool kPbc[3];

__global__ void minimum_image_smith_method(const int num_spins,
    const jams::Real r_ix, const jams::Real r_iy, const jams::Real r_iz, const jams::Real* r, jams::Real* r_ij_out) {

  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < num_spins) {

  jams::Real r_ij[3] = {
      r_ix - r[3*j + 0],
      r_iy - r[3*j + 1],
      r_iz - r[3*j + 2]};


  // [0][0] = 0, [0][1] = 1, [0][2] = 2
  // [1][0] = 3, [1][1] = 4, [1][2] = 5
  // [2][0] = 6, [2][1] = 7, [2][2] = 8

  jams::Real s_ij[3] = {
      kMatrixInv[0] * r_ij[0] + kMatrixInv[1] * r_ij[1] + kMatrixInv[2] * r_ij[2],
      kMatrixInv[3] * r_ij[0] + kMatrixInv[4] * r_ij[1] + kMatrixInv[5] * r_ij[2],
      kMatrixInv[6] * r_ij[0] + kMatrixInv[7] * r_ij[1] + kMatrixInv[8] * r_ij[2]};

    for (auto n = 0; n < 3; ++n) {
      if (kPbc[n]) {
        s_ij[n] = s_ij[n] - trunc(2.0 * s_ij[n]);
      }
    }

    r_ij_out[3*j + 0] = kMatrix[0] * s_ij[0] + kMatrix[1] * s_ij[1] + kMatrix[2] * s_ij[2];
    r_ij_out[3*j + 1] = kMatrix[3] * s_ij[0] + kMatrix[4] * s_ij[1] + kMatrix[5] * s_ij[2];
    r_ij_out[3*j + 2] = kMatrix[6] * s_ij[0] + kMatrix[7] * s_ij[1] + kMatrix[8] * s_ij[2];
  }
}


void jams::cuda_minimum_image(const Vec3R &a, const Vec3R &b, const Vec3R &c, const Vec3b &pbc,
                   const Vec3R &r_i, const jams::MultiArray<jams::Real,2>& r, jams::MultiArray<jams::Real,2>& r_ij) {
  assert(r.size(1) == 3 && r_ij.size(1) == 3);
  assert(r.size(0) == r_ij.size(0));

  static Vec3R a_cached = {0, 0, 0};
  static Vec3R b_cached = {0, 0, 0};
  static Vec3R c_cached = {0, 0, 0};
  static Vec3b pbc_cached = {false, false, false};

  if (a != a_cached || b != b_cached || c != c_cached || pbc_cached != pbc) {
    auto w = matrix_from_cols(a, b, c);
    auto w_inv = inverse(w);
    cudaMemcpyToSymbol(kMatrix, w.data(), 9 * sizeof(&w[0][0]));
    cudaMemcpyToSymbol(kMatrixInv, w_inv.data(), 9 * sizeof(&w_inv[0][0]));
    cudaMemcpyToSymbol(kPbc, pbc.data(), 3 * sizeof(bool));
  }

  dim3 block_size = {64, 1, 1};
  dim3 grid_size = {(int(r.size(0)) + block_size.x - 1) / block_size.x, 1, 1};

  minimum_image_smith_method<<<grid_size, block_size>>>(
    r.size(0), r_i[0], r_i[1], r_i[2], r.device_data(), r_ij.device_data());
}