#ifndef JAMS_SOLVERS_LLG_SOLVER_UTILS_H
#define JAMS_SOLVERS_LLG_SOLVER_UTILS_H

#include <cmath>

#include "jams/containers/vec3.h"

namespace jams::solvers {

inline Vec3 tangent_projection(const Vec3& vector, const Vec3& spin) {
  return vector - jams::dot(vector, spin) * spin;
}

inline Vec3 llg_rhs(const Vec3& spin,
                    const Vec3& field,
                    const double gyro,
                    const double alpha,
                    const Vec3& extra_torque = {0.0, 0.0, 0.0},
                    const double mu = 1.0) {
  const auto spin_cross_field = jams::cross(spin, field);
  return -gyro * (spin_cross_field
                  + alpha * jams::cross(spin, spin_cross_field)
                  + (1.0 / mu) * jams::cross(spin, jams::cross(spin, extra_torque)));
}

inline Vec3 llg_omega(const Vec3& spin,
                      const Vec3& field,
                      const double gyro,
                      const double alpha,
                      const Vec3& extra_torque = {0.0, 0.0, 0.0},
                      const double mu = 1.0) {
  return gyro * (field + alpha * jams::cross(spin, field) + (1.0 / mu) * jams::cross(spin, extra_torque));
}

inline Vec3 cayley_rotate(const Vec3& phi, const Vec3& spin) {
  const auto phi_cross_spin = jams::cross(phi, spin);
  const auto norm_sq = jams::norm_squared(phi);
  const double scale = 1.0 / (1.0 + 0.25 * norm_sq);
  return spin + scale * (phi_cross_spin + 0.5 * jams::cross(phi, phi_cross_spin));
}

inline Vec3 rodrigues_rotate(const Vec3& phi, const Vec3& spin) {
  const double theta_sq = jams::norm_squared(phi);
  const auto cross1 = jams::cross(phi, spin);
  const auto cross2 = jams::cross(phi, cross1);

  if (theta_sq < 1e-8) {
    const double theta_four = theta_sq * theta_sq;
    const double a = 1.0 - theta_sq / 6.0 + theta_four / 120.0;
    const double b = 0.5 - theta_sq / 24.0 + theta_four / 720.0;
    return spin + a * cross1 + b * cross2;
  }

  const double theta = std::sqrt(theta_sq);
  return spin + (std::sin(theta) / theta) * cross1
      + ((1.0 - std::cos(theta)) / theta_sq) * cross2;
}

inline Vec3 dexp_inv_so3(const Vec3& phi, const Vec3& vector) {
  const double theta_sq = jams::norm_squared(phi);
  const auto cross1 = jams::cross(phi, vector);
  const auto cross2 = jams::cross(phi, cross1);

  if (theta_sq < 1e-8) {
    const double theta_four = theta_sq * theta_sq;
    const double beta = (1.0 / 12.0) + theta_sq / 720.0 + theta_four / 30240.0;
    return vector - 0.5 * cross1 + beta * cross2;
  }

  const double theta = std::sqrt(theta_sq);
  const double half_theta = 0.5 * theta;
  const double beta = (1.0 / theta_sq) * (1.0 - half_theta * std::cos(half_theta) / std::sin(half_theta));
  return vector - 0.5 * cross1 + beta * cross2;
}

}  // namespace jams::solvers

#endif  // JAMS_SOLVERS_LLG_SOLVER_UTILS_H
