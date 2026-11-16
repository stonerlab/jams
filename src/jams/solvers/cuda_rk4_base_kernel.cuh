#ifndef JAMS_SOLVERS_CUDA_RK4_BASE_KERNEL_CUH
#define JAMS_SOLVERS_CUDA_RK4_BASE_KERNEL_CUH

namespace {

__global__ void cuda_rk4_mid_step_kernel(const unsigned int num_components, const double stage_dt,
                                         const double* initial, const double* slope,
                                         double* state) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_components) return;
  state[index] = initial[index] + stage_dt * slope[index];
}

__global__ void cuda_rk4_combination_kernel(double* state, const double* initial, const double* k1,
                                            const double* k2, const double* k3, const double* k4,
                                            const double dt, const unsigned int num_components) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_components) return;
  state[index] =
      initial[index] + dt * (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index]) / 6.0;
}

}  // namespace

#endif  // JAMS_SOLVERS_CUDA_RK4_BASE_KERNEL_CUH
