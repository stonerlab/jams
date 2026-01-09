#ifndef JAMS_CORE_CUDA_VECTOR_OPS_H
#define JAMS_CORE_CUDA_VECTOR_OPS_H

template <typename T>
__device__ __forceinline__
T pow_int(T x, int p)
{
	T r = T(1.0);
#pragma unroll
	for (int i = 0; i < p; ++i)
		r *= x;
	return r;
}

__device__ __forceinline__ float dot(const float3 a, const float3 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __forceinline__ double dot(const double3 a, const double3 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __forceinline__ float dot(const float v1[3], const float v2[3]) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__device__ __forceinline__ double dot(const double v1[3], const double v2[3]) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__device__ __forceinline__ double dot(const double3 &a, const double3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Returns the square of the Euclidean norm (x^2 + y^2 + z^2) of the vector.
__device__ __forceinline__ float norm_squared(const float v1[3]) {
	return dot(v1, v1);
}

/// Returns the square of the Euclidean norm (x^2 + y^2 + z^2) of the vector.
__device__ __forceinline__ double norm_squared(const double v1[3]) {
	return dot(v1, v1);
}

/// Returns the Euclidean norm (x^2 + y^2 + z^2) of the vector.
__device__ __forceinline__ double norm(const double v1[3]) {
  return norm3d(v1[0], v1[1], v1[2]);
}

/// Returns the reciprocal of the Euclidean norm 1/(x^2 + y^2 + z^2) of the vector.
__device__ __forceinline__ double rnorm(const double v1[3]) {
	return rnorm3d(v1[0], v1[1], v1[2]);
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

__host__ __device__ __forceinline__
double cross_product_x(const double a[3], const double b[3]) {
  return a[1] * b[2] - a[2] * b[1];
}

__host__ __device__ __forceinline__
double cross_product_y(const double a[3], const double b[3]) {
  return a[2] * b[0] - a[0] * b[2];
}

__host__ __device__ __forceinline__
double cross_product_z(const double a[3], const double b[3]) {
  return a[0] * b[1] - a[1] * b[0];
}

__host__ __device__
inline void cross_product(const double a[3], const double b[3], double c[3]) {
  c[0] = cross_product_x(a, b);
  c[1] = cross_product_y(a, b);
  c[2] = cross_product_z(a, b);
}

__host__ __device__ __forceinline__
double3 cross_product(const double3 a, const double3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

// a.(b x c)
__host__ __device__
inline double scalar_triple_product(const double a[3], const double b[3], const double c[3]) {
  return a[0] * cross_product_x(b, c)
         + a[1]* cross_product_y(b, c)
         + a[2]* cross_product_z(b, c);
}

#endif // JAMS_CORE_CUDA_VECTOR_OPS_H
