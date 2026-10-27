#ifndef JAMS_SOLVERS_LLG_SPIN_RATE_H
#define JAMS_SOLVERS_LLG_SPIN_RATE_H

#include <cmath>
#include "jams/helpers/finite.h"

namespace jams::solvers {

// Field is -dE/dS in meV; inv_mu converts it to Tesla.
// Return ps^-1, or -1 for invalid input/result (propagated by both reductions).
template <typename Real>
JAMS_HOST_DEVICE inline double llg_spin_rate(
    const double* s, const Real* field, const double inv_mu,
    const double alpha, const double gyro_eff) {
  if (!jams::is_finite(inv_mu) || inv_mu < 0.0) return -1.0;
  if (inv_mu == 0.0) return 0.0;  // vacancy
  if (!jams::is_finite(alpha) || !jams::is_finite(gyro_eff)) return -1.0;
  double h[3];
  for (int j = 0; j < 3; ++j) {
    if (!jams::is_finite(s[j])) return -1.0;
    h[j] = field ? static_cast<double>(field[j]) * inv_mu : 0.0;
    if (!jams::is_finite(h[j])) return -1.0;
  }
  const double c[3] = {
      s[1] * h[2] - s[2] * h[1],
      s[2] * h[0] - s[0] * h[2],
      s[0] * h[1] - s[1] * h[0]};
  const double d[3] = {
      s[1] * c[2] - s[2] * c[1],
      s[2] * c[0] - s[0] * c[2],
      s[0] * c[1] - s[1] * c[0]};
  double norm_sq = 0.0;
  for (int j = 0; j < 3; ++j) {
    const double rhs = -gyro_eff * (c[j] + alpha * d[j]);
    if (!jams::is_finite(rhs)) return -1.0;
    norm_sq += rhs * rhs;
  }
  if (!jams::is_finite(norm_sq)) return -1.0;
  return sqrt(norm_sq);
}

JAMS_HOST_DEVICE inline double combine_spin_rates(double a, double b) {
  return (a < 0.0 || b < 0.0) ? -1.0 : (a > b ? a : b);
}

}  // namespace jams::solvers
#endif
