// spinops.cc                                                          -*-C++-*-
#include <jams/helpers/spinops.h>

void jams::rotate_spins(jams::MultiArray<double, 2> &spins,
                        const Mat3 &rotation_matrix) {
  for (auto i = 0; i < spins.size(0); ++i) {
    Vec3 spin = rotation_matrix * Vec3{spins(i,0), spins(i,1), spins(i,2)};
    for (auto j = 0; j < 3; ++j) {
      spins(i, j) = spin[j];
    }
  }
}


void jams::rotate_spins(jams::MultiArray<double, 2> &spins,
                        const Mat3 &rotation_matrix,
                        const jams::MultiArray<int, 1> &indices) {
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    Vec3 spin = rotation_matrix * Vec3{spins(i,0), spins(i,1), spins(i,2)};
    for (auto j = 0; j < 3; ++j) {
      spins(i, j) = spin[j];
    }
  }
}


Vec3 jams::sum_spins(const jams::MultiArray<double, 2> &spins,
                     const jams::MultiArray<int, 1> &indices) {
  Vec3 sum = {0.0, 0.0, 0.0};
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    for (auto j = 0; j < 3; ++j) {
      sum[j] += spins(i, j);
    }
  }

  return sum;
}


Vec3 jams::sum_spins_moments(const jams::MultiArray<double, 2> &spins,
                             const jams::MultiArray<double, 1> &mus) {
  Vec3 sum = {0.0, 0.0, 0.0};
  for (auto i = 0; i < spins.size(0); ++i) {
    for (auto j = 0; j < 3; ++j) {
      sum[j] += mus(i) * spins(i, j);
    }
  }

  return sum;
}


Vec3 jams::sum_spins_moments(const jams::MultiArray<double, 2> &spins,
                             const jams::MultiArray<double, 1> &mus,
                     const jams::MultiArray<int, 1> &indices) {
  Vec3 sum = {0.0, 0.0, 0.0};
  for (auto n = 0; n < indices.size(); ++n) {
    auto i = indices(n);
    for (auto j = 0; j < 3; ++j) {
      sum[j] += mus(i) * spins(i, j);
    }
  }

  return sum;
}

