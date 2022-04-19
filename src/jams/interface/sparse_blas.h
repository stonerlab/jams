//
// Created by Joseph Barker on 2019-10-10.
//

#ifndef JAMS_INTERFACE_SPARSE_BLAS_H
#define JAMS_INTERFACE_SPARSE_BLAS_H

#include <cstring>

#include "jams/interface/openmp.h"

namespace jams {
        template<typename MatType, typename VecType>
        [[gnu::hot]]
        VecType Xcsrmv_general_row(
            const MatType *csr_val,
            const int *csr_col,
            const int *csr_row,
            const VecType *x,
            const int row) {

          VecType sum = 0.0;
          for (auto j = csr_row[row]; j < csr_row[row + 1]; ++j) {
            sum += x[csr_col[j]] * csr_val[j];
          }
          return sum;
        }

        template<typename MatType, typename VecType>
        VecType Xcoomv_general_row(
            const MatType *coo_val,
            const int *coo_col,
            const int *coo_row,
            const VecType *x,
            const int row) {

          // coordinates must be sorted by row

          auto i = 0;
          // skip through until we hit the row of interest
          while (coo_row[i] < row) {
            ++i;
          }

          // process just the rows of interest and then finish
          VecType sum = 0.0;
          while(coo_row[i] == row) {
            auto col = coo_col[i];
            auto val = coo_val[i];
            sum += x[col] * val;
            ++i;
          }

          return sum;
        }

        template<typename MatType, typename VecType>
        [[gnu::hot]]
        void Xcsrmv_general(
            const VecType &alpha,
            const VecType &beta,
            const int &m,
            const MatType *csr_val,
            const int *csr_col,
            const int *csr_row,
            const VecType *x,
            double *y) {

          if (alpha == 1.0 && beta == 0.0) {
            OMP_PARALLEL_FOR
            for (auto i = 0; i < m; ++i) {  // iterate num_rows
              y[i] = Xcsrmv_general_row(csr_val, csr_col, csr_row, x, i);
            }
          } else {
            OMP_PARALLEL_FOR
            for (auto i = 0; i < m; ++i) {  // iterate num_rows
              auto sum = Xcsrmv_general_row(csr_val, csr_col, csr_row, x, i);
              y[i] = beta * y[i] + alpha * sum;
            }
          }
        }

        template<typename MatType, typename VecType>
        [[gnu::hot]]
        void Xcoomv_general(
            const VecType &alpha,
            const VecType &beta,
            const int &m,
            const int &nnz,
            const MatType *coo_val,
            const int *coo_col,
            const int *coo_row,
            const VecType *x,
            double *y) {

          if (alpha == 1.0 && beta == 0.0) {
            memset(y, 0.0, sizeof(double)*m);
            OMP_PARALLEL_FOR
            for (auto i = 0; i < nnz; ++i) {
              auto row = coo_row[i];
              auto col = coo_col[i];
              auto val = coo_val[i];
              y[row] += x[col] * val;
            }
          } else {
            OMP_PARALLEL_FOR
            for (auto i = 0; i < m; ++i) {
              y[i] *= beta;
            }
            OMP_PARALLEL_FOR
            for (auto i = 0; i < nnz; ++i) {
              auto row = coo_row[i];
              auto col = coo_col[i];
              auto val = coo_val[i];
              y[row] += alpha * x[col] * val;
            }
          }
        }
}

#endif //JAMS_INTERFACE_SPARSE_BLAS_H
