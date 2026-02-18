//
// Created by Codex on 2026-02-17.
//

#ifndef JAMS_INTERFACE_LAPACK_TRIDIAGONAL_H
#define JAMS_INTERFACE_LAPACK_TRIDIAGONAL_H

#include <vector>

namespace jams {

// Solve a symmetric tridiagonal eigensystem and return the top-k eigenvectors.
// `diag` and `off` are modified by LAPACK.
void solve_symmetric_tridiagonal_top_eigenvectors(std::vector<double>& diag,
                                                  std::vector<double>& off,
                                                  std::vector<double>& eigenvectors,
                                                  std::vector<double>& eigenvalues,
                                                  int n,
                                                  int k);

} // namespace jams

#endif // JAMS_INTERFACE_LAPACK_TRIDIAGONAL_H
