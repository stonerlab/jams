
__global__ void cuda_reduced_mz_kernel(const int num_spins, const double mx, const double my, const double mz, double * derivs) {

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_spins) {
    const double m = sqrt(mx*mx + my*my + mz*mz);
    derivs[3*idx + 0] = -(1/num_spins) * mx*mz / (m*m*m);
    derivs[3*idx + 1] = -(1/num_spins) * my*mz / (m*m*m);
    derivs[3*idx + 2] = (1/num_spins) * (m*m - mz*mz) / (m*m*m);
  }
}

__global__ void cuda_reduced_mz_kernel2(const int num_spins, const double dx, const double dy, const double dz, double * derivs) {

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_spins) {
    derivs[3*idx + 0] = dx;
    derivs[3*idx + 1] = dy;
    derivs[3*idx + 2] = dz;
  }
}