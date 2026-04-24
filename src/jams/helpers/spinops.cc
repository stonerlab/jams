// spinops.cc                                                          -*-C++-*-
#include <jams/helpers/spinops.h>

void jams::rotate_spins(jams::MultiArray<double, 2> &spins,
                        const Mat<double, 3, 3> &rotation_matrix) {
  for (auto i = 0; i < spins.extent(0); ++i) {
    Vec<double, 3> spin = rotation_matrix * Vec<double, 3>{spins(i,0), spins(i,1), spins(i,2)};
    for (auto j = 0; j < 3; ++j) {
      spins(i, j) = spin[j];
    }
  }
}


void jams::rotate_spins(jams::MultiArray<double, 2> &spins,
                        const Mat<double, 3, 3> &rotation_matrix,
                        const jams::MultiArray<int, 1> &indices) {
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    Vec<double, 3> spin = rotation_matrix * Vec<double, 3>{spins(i,0), spins(i,1), spins(i,2)};
    for (auto j = 0; j < 3; ++j) {
      spins(i, j) = spin[j];
    }
  }
}


Vec<double, 3> jams::sum_spins(const jams::MultiArray<double, 2> &spins,
                     const jams::MultiArray<int, 1> &indices) {
  Vec<double, 3> sum = {0.0, 0.0, 0.0};
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    for (auto j = 0; j < 3; ++j) {
      sum[j] += spins(i, j);
    }
  }

  return sum;
}


Vec<double, 3> jams::sum_spins_moments(const jams::MultiArray<double, 2> &spins,
                             const jams::MultiArray<jams::Real, 1> &mus) {
  Vec<double, 3> sum = {0.0, 0.0, 0.0};
  for (auto i = 0; i < spins.extent(0); ++i) {
    for (auto j = 0; j < 3; ++j) {
      sum[j] += mus(i) * spins(i, j);
    }
  }

  return sum;
}


Vec<double, 3> jams::sum_spins_moments(const jams::MultiArray<double, 2> &spins,
                             const jams::MultiArray<jams::Real, 1> &mus,
                     const jams::MultiArray<int, 1> &indices) {
  Vec<double, 3> sum = {0.0, 0.0, 0.0};
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    for (auto j = 0; j < 3; ++j) {
      sum[j] += mus(i) * spins(i, j);
    }
  }

  return sum;
}

