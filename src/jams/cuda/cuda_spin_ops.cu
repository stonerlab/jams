#include <jams/cuda/cuda_spin_ops.h>
#include <jams/cuda/cuda_device_vector_ops.h>

__global__ void cuda_normalise_spins_kernel(double * spins, const unsigned size)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    double s[3] = {spins[3*idx + 0], spins[3*idx + 1], spins[3*idx + 2]};

    double recip_snorm = rsqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);

    for (auto n = 0; n < 3; ++n) {
      spins[3*idx + n] = s[n] * recip_snorm;
    }
  }
}

void jams::normalise_spins_cuda(jams::MultiArray<double, 2> &spins) {
  dim3 block_size;
  block_size.x = 128;

  dim3 grid_size;
  grid_size.x = (spins.size(0) + block_size.x - 1) / block_size.x;

  cuda_normalise_spins_kernel<<<grid_size, block_size>>>(spins.device_data(), spins.size(0));
}

__global__ void cuda_rotate_spins_kernel(double* spins, const int* indices, const unsigned size,
                                         double Rxx, double Rxy, double Rxz,
                                         double Ryx, double Ryy, double Ryz,
                                         double Rzx, double Rzy, double Rzz) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    double s[3] = {spins[3*indices[idx] + 0], spins[3*indices[idx] + 1], spins[3*indices[idx] + 2]};

    spins[3*indices[idx] + 0] = Rxx * s[0] + Rxy * s[1] + Rxz * s[2];
    spins[3*indices[idx] + 1] = Ryx * s[0] + Ryy * s[1] + Ryz * s[2];
    spins[3*indices[idx] + 2] = Rzx * s[0] + Rzy * s[1] + Rzz * s[2];
  }

}

void jams::rotate_spins_cuda(jams::MultiArray<double, 2> &spins,
                             const Mat3 &rotation_matrix,
                             const jams::MultiArray<int, 1> &indices) {

  dim3 block_size;
  block_size.x = 128;

  dim3 grid_size;
  grid_size.x = (indices.size() + block_size.x - 1) / block_size.x;

  cuda_rotate_spins_kernel<<<grid_size, block_size>>>(
      spins.device_data(), indices.device_data(), indices.size(),
      rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2],
      rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2],
      rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2]);
}


__global__ void cuda_scale_spins_kernel(double* spins, const int* indices, const unsigned size,
                                         const double scale_factor) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    spins[3*indices[idx] + 0] *= scale_factor;
    spins[3*indices[idx] + 1] *= scale_factor;
    spins[3*indices[idx] + 2] *= scale_factor;
  }

}

void jams::scale_spins_cuda(jams::MultiArray<double, 2> &spins,
                             const double &scale_factor,
                             const jams::MultiArray<int, 1> &indices) {

  dim3 block_size;
  block_size.x = 128;

  dim3 grid_size;
  grid_size.x = (indices.size() + block_size.x - 1) / block_size.x;

  cuda_scale_spins_kernel<<<grid_size, block_size>>>(
      spins.device_data(), indices.device_data(), indices.size(), scale_factor);
}


__global__ void cuda_add_to_spin_length_kernel(double* spins, const int* indices, const unsigned size,
                                               const double additional_length) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    double s[3] = {spins[3*indices[idx] + 0], spins[3*indices[idx] + 1], spins[3*indices[idx] + 2]};

    double s_norm = norm(s);

    // If the length of s is zero then it has no direction and we don't add
    // anything to the length.
    if (s_norm != 0.0) {
      spins[3 * indices[idx] + 0] += additional_length * s[0] / s_norm;
      spins[3 * indices[idx] + 1] += additional_length * s[1] / s_norm;
      spins[3 * indices[idx] + 2] += additional_length * s[2] / s_norm;
    }
  }

}

void jams::add_to_spin_length_cuda(jams::MultiArray<double, 2> &spins,
                                   const double &additional_length,
                                   const jams::MultiArray<int, 1> &indices) {

  dim3 block_size;
  block_size.x = 128;

  dim3 grid_size;
  grid_size.x = (indices.size() + block_size.x - 1) / block_size.x;

  cuda_add_to_spin_length_kernel<<<grid_size, block_size>>>(
      spins.device_data(), indices.device_data(), indices.size(),
      additional_length);
}

__global__ void cuda_add_to_spin_kernel(double* spins, const int* indices, const unsigned size,
                                               const double additional_length_x, const double additional_length_y, const double additional_length_z) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    spins[3*indices[idx] + 0] += additional_length_x;
    spins[3*indices[idx] + 1] += additional_length_y;
    spins[3*indices[idx] + 2] += additional_length_z;
  }

}

void jams::add_to_spin_cuda(jams::MultiArray<double, 2> &spins,
                                   const Vec3 &additional_length,
                                   const jams::MultiArray<int, 1> &indices) {

  dim3 block_size;
  block_size.x = 128;

  dim3 grid_size;
  grid_size.x = (indices.size() + block_size.x - 1) / block_size.x;

  cuda_add_to_spin_kernel<<<grid_size, block_size>>>(
      spins.device_data(), indices.device_data(), indices.size(),
      additional_length[0], additional_length[1], additional_length[2]);
}