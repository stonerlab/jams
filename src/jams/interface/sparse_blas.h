//
// Created by Joseph Barker on 2019-10-10.
//

#ifndef JAMS_INTERFACE_SPARSE_BLAS_H
#define JAMS_INTERFACE_SPARSE_BLAS_H

#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "jams/interface/openmp.h"

namespace jams {
        template<typename MatType, typename VecType>
        [[gnu::hot, gnu::always_inline]]
        inline VecType Xcsrmv_general_row(
            const MatType * __restrict csr_val,
            const int     * __restrict csr_col,
            const int     * __restrict csr_row,
            const VecType * __restrict x,
            const int row) {

          const int start = csr_row[row];
          const int end   = csr_row[row + 1];
          const int len   = end - start;

          if (len <= 0) {
            return static_cast<VecType>(0.0);
          }

          const MatType * __restrict val = csr_val + start;
          const int     * __restrict col = csr_col + start;

          VecType sum = static_cast<VecType>(0.0);

          // Hint the compiler to vectorise; gather may still limit speed but this
          // helps on compilers with OpenMP SIMD support.
          #if defined(_OPENMP)
          #pragma omp simd reduction(+:sum)
          #endif
          for (int k = 0; k < len; ++k) {
            const VecType vx = x[col[k]];
            #if defined(__FMA__) || defined(__FMA4__) || defined(__AVX2__)
              sum = std::fma(static_cast<VecType>(val[k]), vx, sum);
            #else
              sum += vx * static_cast<VecType>(val[k]);
            #endif
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
            const MatType * __restrict csr_val,
            const int * __restrict csr_col,
            const int * __restrict csr_row,
            const VecType * __restrict x,
            double * __restrict y) {

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

          // Build row boundaries (row_ptr) from COO rows in O(nnz) by a single scan.
          // Assumes coo_row is sorted by row.
          std::vector<int> row_ptr;
          row_ptr.resize(m + 1);
          int k = 0;
          for (int r = 0; r < m; ++r) {
            row_ptr[r] = k;
            while (k < nnz && coo_row[k] == r) { ++k; }
          }
          row_ptr[m] = k; // should equal nnz

          if (alpha == static_cast<VecType>(1.0) && beta == static_cast<VecType>(0.0)) {
            // y = A * x (COO), rows are independent
            std::fill_n(y, m, 0.0);
            OMP_PARALLEL_FOR
            for (int r = 0; r < m; ++r) {
              const int start = row_ptr[r];
              const int end   = row_ptr[r + 1];
              double sum = 0.0;
              #if defined(_OPENMP)
              #pragma omp simd reduction(+:sum)
              #endif
              for (int j = start; j < end; ++j) {
                const auto vx = x[coo_col[j]];
                #if defined(__FMA__) || defined(__FMA4__) || defined(__AVX2__)
                  sum = std::fma(static_cast<double>(coo_val[j]), static_cast<double>(vx), sum);
                #else
                  sum += static_cast<double>(vx) * static_cast<double>(coo_val[j]);
                #endif
              }
              y[r] = sum;
            }
          } else {
            // y = beta*y + alpha*A*x (COO)
            OMP_PARALLEL_FOR
            for (int r = 0; r < m; ++r) {
              y[r] *= static_cast<double>(beta);
            }
            OMP_PARALLEL_FOR
            for (int r = 0; r < m; ++r) {
              const int start = row_ptr[r];
              const int end   = row_ptr[r + 1];
              double sum = 0.0;
              #if defined(_OPENMP)
              #pragma omp simd reduction(+:sum)
              #endif
              for (int j = start; j < end; ++j) {
                const auto vx = x[coo_col[j]];
                #if defined(__FMA__) || defined(__FMA4__) || defined(__AVX2__)
                  sum = std::fma(static_cast<double>(coo_val[j]), static_cast<double>(vx), sum);
                #else
                  sum += static_cast<double>(vx) * static_cast<double>(coo_val[j]);
                #endif
              }
              y[r] += static_cast<double>(alpha) * sum;
            }
          }
        }
}

#endif //JAMS_INTERFACE_SPARSE_BLAS_H
