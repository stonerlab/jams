// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_FIELDS_H
#define JAMS_CORE_FIELDS_H

#include "core/sparsematrix.h"

#include "jblib/containers/array.h"

void ComputeEffectiveFields();

void ComputeBilinearScalarInteractions(
    const SparseMatrix<float>& interaction_matrix, const jblib::Array<double, 2>&
    spins, jblib::Array<double, 2>* fields);

#ifdef CUDA
void calc_scalar_bilinear(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#else
void calc_scalar_bilinear(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#endif
#ifdef CUDA
void calc_scalar_biquadratic(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#else
void calc_scalar_biquadratic(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#endif
#ifdef CUDA
void calc_tensor_biquadratic(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#else
void calc_tensor_biquadratic(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#endif
#ifdef CUDA
void calc_tensor_bilinear(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#else
void calc_tensor_bilinear(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#endif
#endif  // JAMS_CORE_FIELDS_H
