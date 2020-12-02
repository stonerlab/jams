#ifndef JAMS_CORE_CUDA_VECTOR_OPS_H
#define JAMS_CORE_CUDA_VECTOR_OPS_H

__device__ inline double pow2(double x) {
  return x * x;
}

__device__ inline float dot(const float v1[3], const float v2[3]) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__device__ inline double dot(const double v1[3], const double v2[3]) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__device__ inline double dot(const double3 &a, const double3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float abs(const float v1[3]) {
	return dot(v1, v1);
}

__device__ inline double abs(const double v1[3]) {
	return dot(v1, v1);
}

__device__ inline void matmul(const float mat[3][3], const float v_in[3], float v_out[3]) {
	v_out[0] = mat[0][0] * (v_in[0])
	         + mat[0][1] * (v_in[1])
	         + mat[0][2] * (v_in[2]);

	v_out[1] = mat[1][0] * (v_in[0])
	         + mat[1][1] * (v_in[1])
	         + mat[1][2] * (v_in[2]);

	v_out[2] = mat[2][0] * (v_in[0])
	         + mat[2][1] * (v_in[1])
	         + mat[2][2] * (v_in[2]);
}

__device__ inline void matmul(const double mat[3][3], const double v_in[3], double v_out[3]) {
	v_out[0] = mat[0][0] * (v_in[0])
	         + mat[0][1] * (v_in[1])
	         + mat[0][2] * (v_in[2]);

	v_out[1] = mat[1][0] * (v_in[0])
	         + mat[1][1] * (v_in[1])
	         + mat[1][2] * (v_in[2]);

	v_out[2] = mat[2][0] * (v_in[0])
	         + mat[2][1] * (v_in[1])
	         + mat[2][2] * (v_in[2]);
}

__host__ __device__
inline double cross_product_x(const double a[3], const double b[3]) {
  return a[1] * b[2] - a[2] * b[1];
}

__host__ __device__
inline double cross_product_y(const double a[3], const double b[3]) {
  return a[2] * b[0] - a[0] * b[2];
}

__host__ __device__
inline double cross_product_z(const double a[3], const double b[3]) {
  return a[0] * b[1] - a[1] * b[0];
}

__host__ __device__
inline void cross_product(const double a[3], const double b[3], double c[3]) {
  c[0] = cross_product_x(a, b);
  c[1] = cross_product_y(a, b);
  c[2] = cross_product_z(a, b);
}

// a.(b x c)
__host__ __device__
inline double scalar_triple_product(const double a[3], const double b[3], const double c[3]) {
  return a[0] * cross_product_x(b, c)
         + a[1]* cross_product_y(b, c)
         + a[2]* cross_product_z(b, c);
}

#endif // JAMS_CORE_CUDA_VECTOR_OPS_H
