// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/fields.h"

#include "core/globals.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif

void compute_bilinear_scalar_interactions(
    const SparseMatrix<float>& interaction_matrix, const jblib::Array<double, 2>&
    spin, jblib::Array<double, 2>* field) {
  using globals::num_spins;

  register int i, j, k, l, begin, end;

  for (i = 0; i < num_spins; ++i) {  // iterate rows
    begin = interaction_matrix.row(i);
    end = interaction_matrix.row(i+1);
    for (j = begin; j < end; ++j) {
      k = interaction_matrix.col(j);  // column
      // upper triangle and diagonal
      if (i > (k-1)) {
        for (l = 0; l < 3; ++l) {
          (*field)(i, l) += spin(k, l)*interaction_matrix.val(j);
        }
      }
    }
    for (j = begin; j < end; ++j) {
      k = interaction_matrix.col(j);  // column
      // lower triangle
      if ( i > k ) {
        for (l = 0; l < 3; ++l) {
          (*field)(i, l) += spin(i, l)*interaction_matrix.val(j);
        }
      }
    }
  }
}

#ifdef CUDA
void compute_bilinear_scalar_interactions_csr(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y)
#else
void compute_bilinear_scalar_interactions_csr(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y)
#endif
{
  using namespace globals;
  int i, j, k, l;
  int begin, end;
  for (i = 0; i < num_spins; ++i) {  // iterate rows
    begin = ptrb[i]; end = ptre[i];
    for (j = begin; j < end; ++j) {
      k = indx[j];  // column
      // upper triangle and diagonal
      if (i > (k-1)) {
        for (l = 0; l < 3; ++l) {
          y(i, l) = y(i, l) + s(k, l)*val[j];
        }
      }
    }
    for (j = begin; j < end; ++j) {
      k = indx[j];  // column
      // upper triangle and diagonal
      if ( i > k ) {
        for (l = 0; l < 3; ++l) {
          y(k, l) = y(k, l) + s(i, l)*val[j];
        }
      }
    }
  }
}
#ifdef CUDA
void compute_biquadratic_scalar_interactions_csr(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y)
#else
void compute_biquadratic_scalar_interactions_csr(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y)
#endif
{
  // NOTE: Factor of two is included here for biquadratic terms
  using namespace globals;
  double tmp;
  int i, j, k, l;
  int begin, end;
  for (i = 0; i < num_spins; ++i) {  // iterate rows
    begin = ptrb[i]; end = ptre[i];
    for (j = begin; j < end; ++j) {
      k = indx[j];  // column
      // upper triangle and diagonal
      tmp = (s(i, 0)*s(k, 0) + s(i, 1)*s(k, 1) + s(i, 2)*s(k, 2))*val[j];
      if (i > (k-1)) {
        for (l = 0; l < 3; ++l) {
          y(i, l) = y(i, l) + 2.0*s(k, l)*tmp;
        }
      }
    }
    for (j = begin; j < end; ++j) {
      k = indx[j];  // column
      // upper triangle and diagonal
      tmp = (s(i, 0)*s(k, 0) + s(i, 1)*s(k, 1) + s(i, 2)*s(k, 2))*val[j];
      if (i > k) {
        for (l = 0; l < 3; ++l) {
          y(k, l) = y(k, l) + 2.0*s(i, l)*tmp;
        }
      }
    }
  }
}
#ifdef CUDA
void compute_biquadratic_tensor_interactions_csr(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y)
#else
void compute_biquadratic_tensor_interactions_csr(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y)
#endif
{
  // NOTE: Factor of two is included here for biquadratic terms
  // NOTE: Tensor calculations are added to the existing fields
  using namespace globals;
  // dscrmv below has beta=0.0 -> field array is zeroed
  // exchange
  char transa[1] = {'N'};
  char matdescra[6] = {'S', 'L', 'N', 'C', 'N', 'N'};
#ifdef MKL
  double one = 1.0;
  double two = 2.0;
    mkl_dcsrmv(transa, &num_spins3, &num_spins3, &two, matdescra, val,
        indx, ptrb, ptre, s.data(), &zero, y.data());
#else
    jams_dcsrmv(transa, num_spins3, num_spins3, 2.0, matdescra, val,
        indx, ptrb, ptre, s.data(), 1.0, y.data());
#endif
}
#ifdef CUDA
void compute_bilinear_tensor_interactions_csr(const float *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y)
#else
void compute_bilinear_tensor_interactions_csr(const double *val, const int *indx,
  const int *ptrb, const int *ptre, jblib::Array<double, 2> &y)
#endif
{
  // NOTE: this resets the field array to zero
  using namespace globals;
  char transa[1] = {'N'};
  char matdescra[6] = {'S', 'L', 'N', 'C', 'N', 'N'};
#ifdef MKL
  double one = 1.0;
    mkl_dcsrmv(transa, &num_spins3, &num_spins3, &one, matdescra, val,
        indx, ptrb, ptre, s.data(), &zero, y.data());
#else
    jams_dcsrmv(transa, num_spins3, num_spins3, 1.0, matdescra, val,
        indx, ptrb, ptre, s.data(), 1.0, y.data());
#endif
}
void compute_effective_fields() {

}
