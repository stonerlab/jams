// __global__ void cuda_uniaxial_energy_kernel(
//   const int num_spins, 
//   const int num_mca, 
//   const int * mca_order,
//   const double * mca_value, 
//   const double * dev_s, 
//   double * dev_e) {
//   const int idx = blockIdx.x*blockDim.x+threadIdx.x;
//   if (idx < num_spins) {
//     const double sz = dev_s[3*idx+2];
//     double energy = 0.0;
//     for (int n = 0; n < num_mca; ++n) {
//       energy += mca_value[num_mca * idx + n] * cuda_legendre_poly(sz, mca_order[n]);
//     }
//     dev_e[idx] =  energy;
//   }
// }

__global__ void cuda_demag_field_kernel(
  const int num_spins, 
  const double dx,
  const double dy,
  const double dz,
  const double mx,
  const double my,
  const double mz, 
  const double * dev_mus, 
  double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const int idx3 = 3*idx;

    dev_h[idx3 + 0] = -dx*mx*dev_mus[idx];
    dev_h[idx3 + 1] = -dy*my*dev_mus[idx];
    dev_h[idx3 + 2] = -dz*mz*dev_mus[idx]; 
  }
}