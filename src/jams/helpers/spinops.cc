// spinops.cc                                                          -*-C++-*-
#include <jams/helpers/spinops.h>

void jams::rotate_spins(jams::MultiArray<double, 2> &spins,
                        const jams::Mat<double, 3, 3> &rotation_matrix) {
  for (auto i = 0; i < spins.extent(0); ++i) {
    jams::Vec<double, 3> spin = rotation_matrix * jams::Vec<double, 3>{spins(i,0), spins(i,1), spins(i,2)};
    for (auto j = 0; j < 3; ++j) {
      spins(i, j) = spin[j];
    }
  }
}


void jams::rotate_spins(jams::MultiArray<double, 2> &spins,
                        const jams::Mat<double, 3, 3> &rotation_matrix,
                        const jams::MultiArray<int, 1> &indices) {
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    jams::Vec<double, 3> spin = rotation_matrix * jams::Vec<double, 3>{spins(i,0), spins(i,1), spins(i,2)};
    for (auto j = 0; j < 3; ++j) {
      spins(i, j) = spin[j];
    }
  }
}


jams::Vec<double, 3> jams::sum_spins(const jams::MultiArray<double, 2> &spins,
                     const jams::MultiArray<int, 1> &indices) {
  jams::Vec<double, 3> sum = {0.0, 0.0, 0.0};
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    for (auto j = 0; j < 3; ++j) {
      sum[j] += spins(i, j);
    }
  }

  return sum;
}


jams::Vec<double, 3> jams::sum_spins_moments(const jams::MultiArray<double, 2> &spins,
                             const jams::MultiArray<jams::Real, 1> &mus) {
  jams::Vec<double, 3> sum = {0.0, 0.0, 0.0};
  for (auto i = 0; i < spins.extent(0); ++i) {
    for (auto j = 0; j < 3; ++j) {
      sum[j] += mus(i) * spins(i, j);
    }
  }

  return sum;
}


jams::Vec<double, 3> jams::sum_spins_moments(const jams::MultiArray<double, 2> &spins,
                             const jams::MultiArray<jams::Real, 1> &mus,
                     const jams::MultiArray<int, 1> &indices) {
  jams::Vec<double, 3> sum = {0.0, 0.0, 0.0};
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    for (auto j = 0; j < 3; ++j) {
      sum[j] += mus(i) * spins(i, j);
    }
  }

  return sum;
}

