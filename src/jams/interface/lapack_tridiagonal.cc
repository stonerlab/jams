//
// Created by Codex on 2026-02-17.
//

#include "jams/interface/lapack_tridiagonal.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

extern "C" {
void dstevr_(const char* jobz,
             const char* range,
             const int* n,
             double* d,
             double* e,
             const double* vl,
             const double* vu,
             const int* il,
             const int* iu,
             const double* abstol,
             int* m,
             double* w,
             double* z,
             const int* ldz,
             int* isuppz,
             double* work,
             const int* lwork,
             int* iwork,
             const int* liwork,
             int* info);
}

} // namespace

void jams::solve_symmetric_tridiagonal_top_eigenvectors(std::vector<double>& diag,
                                                         std::vector<double>& off,
                                                         std::vector<double>& eigenvectors,
                                                         const int n,
                                                         const int k)
{
  if (k <= 0 || k > n)
  {
    throw std::runtime_error("Invalid tridiagonal eigensolver eigenvector count");
  }

  if (n == 1)
  {
    eigenvectors.assign(1, 1.0);
    return;
  }

  const char jobz = 'V';
  const char range = 'I';
  const int il = n - k + 1;
  const int iu = n;
  const double vl = 0.0;
  const double vu = 0.0;
  const double abstol = 0.0;
  const int ldz = n;
  int m = 0;
  int info = 0;
  std::vector<double> w(static_cast<std::size_t>(k), 0.0);
  eigenvectors.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(k), 0.0);
  std::vector<int> isuppz(static_cast<std::size_t>(2 * std::max(1, k)), 0);

  int lwork = -1;
  int liwork = -1;
  double work_query = 0.0;
  int iwork_query = 0;

  dstevr_(&jobz,
          &range,
          &n,
          diag.data(),
          off.data(),
          &vl,
          &vu,
          &il,
          &iu,
          &abstol,
          &m,
          w.data(),
          eigenvectors.data(),
          &ldz,
          isuppz.data(),
          &work_query,
          &lwork,
          &iwork_query,
          &liwork,
          &info);

  if (info != 0)
  {
    throw std::runtime_error("LAPACK dstevr workspace query failed");
  }

  lwork = std::max(1, static_cast<int>(work_query));
  liwork = std::max(1, iwork_query);
  std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
  std::vector<int> iwork(static_cast<std::size_t>(liwork), 0);

  info = 0;
  dstevr_(&jobz,
          &range,
          &n,
          diag.data(),
          off.data(),
          &vl,
          &vu,
          &il,
          &iu,
          &abstol,
          &m,
          w.data(),
          eigenvectors.data(),
          &ldz,
          isuppz.data(),
          work.data(),
          &lwork,
          iwork.data(),
          &liwork,
          &info);

  if (info != 0 || m != k)
  {
    throw std::runtime_error("LAPACK dstevr failed");
  }
}
