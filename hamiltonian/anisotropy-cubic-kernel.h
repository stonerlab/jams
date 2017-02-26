__host__ __device__ 
inline double cubic_K1_anisotropy_energy(const double K1, const double sx, const double sy, const double sz) {
  return K1 * ((sx * sx * sy * sy) + (sy * sy * sz * sz) + (sx * sx * sz * sz));
} 
//---------------------------------------------------------------------

__host__ __device__ 
inline double cubic_K2_anisotropy_energy(const double K2, const double sx, const double sy, const double sz) {
  return K2 * sx * sx * sy * sy * sz * sz;
} 

//---------------------------------------------------------------------

__host__ __device__ 
inline double cubic_K1_K2_anisotropy_energy(const double K1, const double K2, const double sx, const double sy, const double sz) {
  return cubic_K1_anisotropy_energy(K1, sx, sy, sz) 
    + cubic_K2_anisotropy_energy(K2, sx, sy, sz);
} 

//---------------------------------------------------------------------

__host__ __device__ 
inline double cubic_K1_anisotropy_field_x(const double K1, const double sx, const double sy, const double sz) {
  return -2.0 * K1 * ((sx * sy * sy) + (sx * sz * sz));
} 

__host__ __device__ 
inline double cubic_K1_anisotropy_field_y(const double K1, const double sx, const double sy, const double sz) {
  return -2.0 * K1 * ((sx * sx * sy) + (sy * sz * sz));
} 

__host__ __device__ 
inline double cubic_K1_anisotropy_field_z(const double K1, const double sx, const double sy, const double sz) {
  return -2.0 * K1 * ((sy * sy * sz) + (sx * sx * sz));
} 


//---------------------------------------------------------------------

__host__ __device__ 
inline double cubic_K2_anisotropy_field_x(const double K2, const double sx, const double sy, const double sz) {
  return -2.0 * K2 * sx * sy * sy * sz * sz;
} 

__host__ __device__ 
inline double cubic_K2_anisotropy_field_y(const double K2, const double sx, const double sy, const double sz) {
  return -2.0 * K2 * sx * sx * sy * sz * sz;
} 

__host__ __device__ 
inline double cubic_K2_anisotropy_field_z(const double K2, const double sx, const double sy, const double sz) {
  return -2.0 * K2 * sx * sx * sy * sy * sz;
} 

//---------------------------------------------------------------------

__host__ __device__ 
inline double cubic_K1_K2_anisotropy_field_x(const double K1, const double K2, const double sx, const double sy, const double sz) {
  return cubic_K1_anisotropy_field_x(K1, sx, sy, sz) 
    + cubic_K2_anisotropy_field_x(K2, sx, sy, sz);
} 

__host__ __device__ 
inline double cubic_K1_K2_anisotropy_field_y(const double K1, const double K2, const double sx, const double sy, const double sz) {
  return cubic_K1_anisotropy_field_y(K1, sx, sy, sz) 
    + cubic_K2_anisotropy_field_y(K2, sx, sy, sz);
} 

__host__ __device__ 
inline double cubic_K1_K2_anisotropy_field_z(const double K1, const double K2, const double sx, const double sy, const double sz) {
  return cubic_K1_anisotropy_field_z(K1, sx, sy, sz) 
    + cubic_K2_anisotropy_field_z(K2, sx, sy, sz);
} 

//---------------------------------------------------------------------


__global__ void cuda_anisotropy_cubic_energy_kernel(const int num_spins, 
  const double * K1_value, const double * K2_value, const double * dev_s, double * dev_e) {
  
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if (idx < num_spins) {
    const double sx = dev_s[3 * idx + 0];
    const double sy = dev_s[3 * idx + 1];
    const double sz = dev_s[3 * idx + 2];

    dev_e[idx] = cubic_K1_K2_anisotropy_energy(K1_value[idx], K2_value[idx], sx, sy, sz); 
  }
}

__global__ void cuda_anisotropy_cubic_field_kernel(const int num_spins, 
  const double * K1_value, const double * K2_value, const double * dev_s, double * dev_h) {
  
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if (idx < num_spins) {
    const double sx = dev_s[3 * idx + 0];
    const double sy = dev_s[3 * idx + 1];
    const double sz = dev_s[3 * idx + 2];

    const double K1 = K1_value[idx];
    const double K2 = K2_value[idx];

    dev_h[3 * idx + 0] = cubic_K1_K2_anisotropy_field_x(K1, K2, sx, sy, sz);
    dev_h[3 * idx + 1] = cubic_K1_K2_anisotropy_field_y(K1, K2, sx, sy, sz);
    dev_h[3 * idx + 2] = cubic_K1_K2_anisotropy_field_z(K1, K2, sx, sy, sz);

  }
}