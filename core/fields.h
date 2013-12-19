#ifndef __FIELDS_H__
#define __FIELDS_H__

#include <containers/array.h>

void calculate_fields();

#ifdef CUDA
void calc_scalar_bilinear(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, jblib::Array<double,2> &y);
#else
void calc_scalar_bilinear(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, jblib::Array<double,2> &y);
#endif

#ifdef CUDA
void calc_scalar_biquadratic(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, jblib::Array<double,2> &y);
#else
void calc_scalar_biquadratic(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, jblib::Array<double,2> &y);
#endif

#ifdef CUDA
void calc_tensor_biquadratic(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, jblib::Array<double,2> &y);
#else
void calc_tensor_biquadratic(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, jblib::Array<double,2> &y);
#endif

#ifdef CUDA
void calc_tensor_bilinear(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, jblib::Array<double,2> &y);
#else
void calc_tensor_bilinear(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, jblib::Array<double,2> &y);
#endif
#endif // __FIELDS_H__
