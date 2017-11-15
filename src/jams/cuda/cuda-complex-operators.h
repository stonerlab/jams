//
// Created by Joe Barker on 2017/05/02.
//
#ifndef JAMS_CUDA_COMPLEX_OPERATORS_H
#define JAMS_CUDA_COMPLEX_OPERATORS_H

#include <cuda_runtime.h>

__host__ __device__
inline cuDoubleComplex operator*(const cuDoubleComplex& v1, const cuDoubleComplex& v2)
{
  return {v1.x * v2.x - v1.y * v2.y,
          v1.x * v2.y + v1.y * v2.x};
}

__host__ __device__
inline cuDoubleComplex operator+(const cuDoubleComplex& v1, const cuDoubleComplex& v2)
{
  return {v1.x + v2.x,
          v1.y + v2.y};
}

__host__ __device__
inline cuDoubleComplex operator*(const double& a, const cuDoubleComplex& v1)
{
  return {a * v1.x,
          a * v1.y};
}

__host__ __device__
inline cuDoubleComplex &operator+=(cuDoubleComplex& v1, const cuDoubleComplex& v2)
{
  v1.x += v2.x;
  v1.y += v2.y;
  return v1;
}




#endif //JAMS_CUDA_COMPLEX_OPERATORS_H
