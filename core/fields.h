// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_FIELDS_H
#define JAMS_CORE_FIELDS_H

#include "core/sparsematrix.h"

#include "jblib/containers/array.h"

void compute_effective_fields();

void ComputeBilinearScalarInteractions(
    const SparseMatrix<float>& interaction_matrix, const jblib::Array<double, 2>&
    spins, jblib::Array<double, 2>* fields);

#ifdef CUDA
void compute_bilinear_scalar_interactions_csr(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#else
void compute_bilinear_scalar_interactions_csr(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#endif
#ifdef CUDA
void compute_biquadratic_scalar_interactions_csr(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#else
void compute_biquadratic_scalar_interactions_csr(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#endif
#ifdef CUDA
void compute_biquadratic_tensor_interactions_csr(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#else
void compute_biquadratic_tensor_interactions_csr(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#endif
#ifdef CUDA
void compute_bilinear_tensor_interactions_csr(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#else
void compute_bilinear_tensor_interactions_csr(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y);
#endif
#endif  // JAMS_CORE_FIELDS_H
