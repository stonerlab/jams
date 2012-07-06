#ifndef __FIELDS_H__
#define __FIELDS_H__

#include "array2d.h"

void calculate_fields();

#ifdef CUDA
void calc_scalar_bilinear(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, Array2D<double> &y);
#else
void calc_scalar_bilinear(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, Array2D<double> &y);
#endif

#ifdef CUDA
void calc_scalar_biquadratic(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, Array2D<double> &y);
#else
void calc_scalar_biquadratic(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, Array2D<double> &y);
#endif

#ifdef CUDA
void calc_tensor_biquadratic(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, Array2D<double> &y);
#else
void calc_tensor_biquadratic(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, Array2D<double> &y);
#endif

#ifdef CUDA
void calc_tensor_bilinear(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, Array2D<double> &y);
#else
void calc_tensor_bilinear(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, Array2D<double> &y);
#endif
#endif // __FIELDS_H__
