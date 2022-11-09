// cvar_reduced_mz_cuda.cu                                             -*-C++-*-
#include <jams/metadynamics/cvars/cvar_reduced_mz_cuda.h>
#include <jams/cuda/cuda_stride_reduce.h>
#include <jams/core/globals.h>

#include <jams/metadynamics/cvars/cvar_reduced_mz_cuda_kernel.cuh>


jams::CVarReducedMzCuda::CVarReducedMzCuda(const libconfig::Setting &settings)
    : CVarReducedMz(settings)
{
  zero(derivatives_.resize(globals::num_spins, 3));
}

double jams::CVarReducedMzCuda::value() {
  Vec3 m;
  for (auto i = 0; i < 3; ++i) {
    m[i] = cuda_stride_reduce_array(globals::s.device_data(), globals::num_spins3, 3, i);
  }

  return m[2] / norm(m);
}


const jams::MultiArray<double, 2>&
jams::CVarReducedMzCuda::derivatives() {
  Vec3 m;
  for (auto i = 0; i < 3; ++i) {
    m[i] = cuda_stride_reduce_array(globals::s.device_data(), globals::num_spins3, 3, i) / (globals::num_spins);
  }

//  // WARNING: THIS WILL BE VERY SLOW. WRITE A KERNEL OR FUNCTION LATER
//  for (auto i = 0; i < globals::num_spins; ++i) {
//    derivatives_(i, 0) = -(1/globals::num_spins) * m[0]*m[2] / pow3(norm(m));
//    derivatives_(i, 1) = -(1/globals::num_spins) * m[1]*m[2] / pow3(norm(m));
//    derivatives_(i, 2) = (1/globals::num_spins) * (pow2(norm(m)) - m[2]*m[2]) / pow3(norm(m));
//  }

  const dim3 block_size = {128, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});


//  cuda_reduced_mz_kernel<<<grid_size, block_size>>>(globals::num_spins, m[0], m[1], m[2], derivatives_.device_data());

  const double dx = -(1.0 / globals::num_spins) * m[0]*m[2] / pow3(norm(m));
  const double dy = -(1.0 / globals::num_spins) * m[1]*m[2] / pow3(norm(m));
  const double dz = -(1.0 / globals::num_spins) * m[2]*m[2] / pow3(norm(m)) + 1.0 / (globals::num_spins * norm(m));

  cuda_reduced_mz_kernel2<<<grid_size, block_size>>>(globals::num_spins, dx, dy, dz, derivatives_.device_data());

  return derivatives_;
}
