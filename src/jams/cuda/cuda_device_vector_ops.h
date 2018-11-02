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

#endif // JAMS_CORE_CUDA_VECTOR_OPS_H
