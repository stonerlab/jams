//
// Created by Joseph Barker on 2018-11-22.
//

#ifndef JAMS_CUDA_SPECTRUM_GENERAL_KERNEL_CUH_H
#define JAMS_CUDA_SPECTRUM_GENERAL_KERNEL_CUH_H

__global__ void CudaSpectrumGeneralKernel(
        const unsigned i,
        const unsigned j,
        const unsigned num_w_points,
        const unsigned num_q_points,
        const cuFloatComplex* qfactors,
        const cuFloatComplex* spin_data,
        cuFloatComplex* spectrum
        )
{
  const unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int q = blockIdx.y * blockDim.y + threadIdx.y;

  if (w < num_w_points && q < num_q_points) {
    auto spin_i = spin_data[num_w_points * i + w];
    auto spin_j = spin_data[num_w_points * j + (num_w_points - w) % num_w_points];
    auto expQR = qfactors[q];

    // this is -kImagOne * qfactors[q] * spin_data_(i,w) * spin_data_(j, (padded_size_ - w) % padded_size_);
    spectrum[num_w_points * q + w].x += -expQR.x * spin_i.x * spin_j.y + expQR.y * spin_i.y * spin_j.x + expQR.x * spin_i.x * spin_j.y + expQR.x * spin_i.y * spin_j.x;
    spectrum[num_w_points * q + w].y += -expQR.x * spin_i.x * spin_j.x - expQR.x * spin_i.y * spin_j.y - expQR.y * spin_i.x * spin_j.y - expQR.y * spin_i.y * spin_j.x;
  }
}

#endif //JAMS_CUDA_SPECTRUM_GENERAL_KERNEL_CUH_H
