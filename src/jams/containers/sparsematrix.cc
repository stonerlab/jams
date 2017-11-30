// Copyright 2014 Joseph Barker. All rights reserved.
#include <cstring>
#include "jams/core/globals.h"

void jams_dcsrmv(const char trans[1], const int m, const int n,
    const double alpha, const char descra[6], const double *val,
    const int *indx, const int *ptrb, const int *ptre, const double *x,
    const double beta, double * y) {
  // symmetric matrix
  int i, j, k;
  int begin, end;
  double tmp;
  if (descra[0] == 'S') {
    // upper matrix
    if (descra[1] == 'L') {

      if (alpha == 1.0 && beta == 0.0) {
        for (i = 0; i < m; ++i) {  // iterate rows
          y[i] = 0.0;
          begin = ptrb[i]; end = ptre[i];
          for (j = begin; j < end; ++j) {  // j is the row
            k = indx[j];  // k is the column
            if (i > k) {  // upper triangle
              y[i] = y[i] + x[k]*val[j];
            }
            if (i > (k-1)) {  // lower triangle and diagonal
              y[k] = y[k] + x[i]*val[j];
            }
          }
        }
      } else {

        for (i = 0; i < m; ++i) {  // iterate rows
          y[i] = beta * y[i];
          begin = ptrb[i]; end = ptre[i];
          for (j = begin; j < end; ++j) {  // j is the row
            k = indx[j];  // k is the column
            tmp = alpha*val[j];
            if (i > k) {  // upper triangle
              y[i] = y[i] + x[k]*tmp;
            }
            if (i > (k-1)) {  // lower triangle and diagonal
              y[k] = y[k] + x[i]*tmp;
            }
          }
        }
      }
    } else if (descra[1] == 'U') {  // lower matrix
      std::cerr << "WARNING: dcsrmv with 'S' and 'U' is untested.\n";
      for (i = 0; i < m; ++i) {  // iterate rows
        y[i] = beta * y[i];
        begin = ptrb[i]; end = ptre[i];
        for (j = begin; j < end; ++j) {
          k = indx[j];  // column
          // lower triangle and diagonal
          tmp = alpha*val[j];
          if (i < (k+1)) {
            y[i] = y[i] + x[k]*tmp;
          }
        }
        for (j = begin; j < end; ++j) {
          k = indx[j];  // column
          // lower triangle and diagonal
          tmp = alpha*val[j];
          if (i < k) {
            y[k] = y[k] + x[i]*tmp;
          }
        }
      }
    }
  // general matrix
  } else {
    if (alpha == 1.0 && beta == 0.0) {
      memset(y, 0, m*sizeof(double));
      for (i = 0; i < m; ++i) {  // iterate rows
        for (j = ptrb[i]; j < ptre[i]; ++j) {
          y[i] = y[i] + x[indx[j]]*val[j];
        }
      }
    } else {
      for (i = 0; i < m; ++i) {  // iterate rows
        y[i] = beta * y[i];
        begin = ptrb[i]; end = ptre[i];
        for (j = begin; j < end; ++j) {
          k = indx[j];  // column
          y[i] = y[i] + alpha*x[k]*val[j];
        }
      }
    }
  }
}

void jams_scsrmv(const char trans[1], const int m, const int n,
    const double alpha, const char descra[6], const float *val,
    const int *indx, const int *ptrb, const int *ptre, const double *x,
    const double beta, double * y) {
  // symmetric matrix
  int i, j, k;
  int begin, end;
  double tmp;
  if (descra[0] == 'S') {
    // upper matrix
    if (descra[1] == 'L') {

      if (alpha == 1.0 && beta == 0.0) {
        for (i = 0; i < m; ++i) {  // iterate rows
          y[i] = 0.0;
          begin = ptrb[i]; end = ptre[i];
          for (j = begin; j < end; ++j) {  // j is the row
            k = indx[j];  // k is the column
            if (i > k) {  // upper triangle
              y[i] = y[i] + x[k]*val[j];
            }
            if (i > (k-1)) {  // lower triangle and diagonal
              y[k] = y[k] + x[i]*val[j];
            }
          }
        }
      } else {

        for (i = 0; i < m; ++i) {  // iterate rows
          y[i] = beta * y[i];
          begin = ptrb[i]; end = ptre[i];
          for (j = begin; j < end; ++j) {  // j is the row
            k = indx[j];  // k is the column
            tmp = alpha*val[j];
            if (i > k) {  // upper triangle
              y[i] = y[i] + x[k]*tmp;
            }
            if (i > (k-1)) {  // lower triangle and diagonal
              y[k] = y[k] + x[i]*tmp;
            }
          }
        }
      }
    } else if (descra[1] == 'U') {  // lower matrix
      std::cerr << "WARNING: dcsrmv with 'S' and 'U' is untested.\n";
      for (i = 0; i < m; ++i) {  // iterate rows
        y[i] = beta * y[i];
        begin = ptrb[i]; end = ptre[i];
        for (j = begin; j < end; ++j) {
          k = indx[j];  // column
          // lower triangle and diagonal
          tmp = alpha*val[j];
          if (i < (k+1)) {
            y[i] = y[i] + x[k]*tmp;
          }
        }
        for (j = begin; j < end; ++j) {
          k = indx[j];  // column
          // lower triangle and diagonal
          tmp = alpha*val[j];
          if (i < k) {
            y[k] = y[k] + x[i]*tmp;
          }
        }
      }
    }
  // general matrix
  } else {
    if (alpha == 1.0 && beta == 0.0) {
      memset(y, 0, m*sizeof(double));
      for (i = 0; i < m; ++i) {  // iterate rows
        for (j = ptrb[i]; j < ptre[i]; ++j) {
          y[i] = y[i] + x[indx[j]]*val[j];
        }
      }
    } else {
      for (i = 0; i < m; ++i) {  // iterate rows
        y[i] = beta * y[i];
        begin = ptrb[i]; end = ptre[i];
        for (j = begin; j < end; ++j) {
          k = indx[j];  // column
          y[i] = y[i] + alpha*x[k]*val[j];
        }
      }
    }
  }
}

