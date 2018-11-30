//
// Created by Joseph Barker on 2018-11-22.
//

#ifndef JAMS_CUDA_SPECTRUM_GENERAL_KERNEL_CUH_H
#define JAMS_CUDA_SPECTRUM_GENERAL_KERNEL_CUH_H

#include "jams/cuda/cuda_device_complex_ops.h"

__global__ void CudaSpectrumGeneralKernel(
        const unsigned i,
        const unsigned j,
        const unsigned num_w_points,
        const unsigned num_q_points,
        const unsigned num_q_vectors,
        const unsigned padded_size,
        const cuFloatComplex* qfactors,
        const cuFloatComplex* spin_data,
        cuFloatComplex* spectrum
        )
{
  const unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (w < num_w_points && q < num_q_points) {
    cuFloatComplex SQw = spectrum[num_w_points * q + w];
    const auto spin_i = spin_data[padded_size * i + w];
    const auto spin_j = cuConjf(spin_data[padded_size * j + (padded_size - w) % padded_size]);
    for (auto n = 0; n < num_q_vectors; ++n) {
      const auto expQR = qfactors[num_q_vectors * q + n];

      // this is -kImagOne * qfactors[q] * spin_data_(i,w) * spin_data_(j, (padded_size_ - w) % padded_size_);
      SQw.x += -(expQR.x * spin_i.x * spin_j.y + expQR.y * spin_i.y * spin_j.x +
                                            expQR.x * spin_i.x * spin_j.y + expQR.x * spin_i.y * spin_j.x);
      SQw.y += -(expQR.x * spin_i.x * spin_j.x - expQR.x * spin_i.y * spin_j.y -
                                            expQR.y * spin_i.x * spin_j.y - expQR.y * spin_i.y * spin_j.x);
    }
    spectrum[num_w_points * q + w] = SQw;
  }
}

#endif //JAMS_CUDA_SPECTRUM_GENERAL_KERNEL_CUH_H
