#ifndef JAMS_HAMILTONIAN_DIPOLE_INTERACTION_H
#define JAMS_HAMILTONIAN_DIPOLE_INTERACTION_H

#include <cmath>
#include <stdexcept>

#include <jams/containers/mat3.h>
#include <jams/containers/sparse_matrix_builder.h>
#include <jams/containers/vec3.h>
#include <jams/helpers/consts.h>
#include <jams/helpers/maths.h>

namespace jams::dipole {

inline double interaction_prefactor(const double lattice_parameter) {
  return kVacuumPermeabilityIU / (4.0 * kPi * pow3(lattice_parameter));
}

template<typename T = double>
inline jams::Mat<T, 3, 3> interaction_tensor(
    const jams::Vec<double, 3>& r_ij,
    const double mu_i,
    const double mu_j,
    const double lattice_parameter,
    const double scale = 1.0) {
  const double r_abs_sq = jams::norm_squared(r_ij);
  if (!std::isnormal(r_abs_sq)) {
    throw std::runtime_error("dipole interaction tensor requires a non-zero finite displacement");
  }

  const double prefactor =
      scale * interaction_prefactor(lattice_parameter) * mu_i * mu_j
      / (r_abs_sq * r_abs_sq * std::sqrt(r_abs_sq));

  jams::Mat<T, 3, 3> tensor{};
  for (auto m = 0; m < 3; ++m) {
    for (auto n = 0; n < 3; ++n) {
      tensor[m][n] = static_cast<T>(
          prefactor * (3.0 * r_ij[m] * r_ij[n]
                       - r_abs_sq * (m == n ? 1.0 : 0.0)));
    }
  }
  return tensor;
}

template<typename TensorType, typename SpinType>
inline auto interaction_field(
    const jams::Mat<TensorType, 3, 3>& interaction,
    const jams::Vec<SpinType, 3>& spin) {
  return interaction * spin;
}

// r_ji must be the current-formula displacement r_j - r_i.
inline void insert_displacement_weighted_interaction(
    jams::SparseMatrix<double>::Builder& rx_builder,
    jams::SparseMatrix<double>::Builder& ry_builder,
    jams::SparseMatrix<double>::Builder& rz_builder,
    const int site_i,
    const int site_j,
    const jams::Vec<double, 3>& r_ji,
    const jams::Mat<double, 3, 3>& interaction) {
  for (auto m = 0; m < 3; ++m) {
    for (auto n = 0; n < 3; ++n) {
      if (interaction[m][n] == 0.0) {
        continue;
      }

      const int row = 3 * site_i + m;
      const int col = 3 * site_j + n;
      if (r_ji[0] != 0.0) {
        rx_builder.insert(row, col, r_ji[0] * interaction[m][n]);
      }
      if (r_ji[1] != 0.0) {
        ry_builder.insert(row, col, r_ji[1] * interaction[m][n]);
      }
      if (r_ji[2] != 0.0) {
        rz_builder.insert(row, col, r_ji[2] * interaction[m][n]);
      }
    }
  }
}

}  // namespace jams::dipole

#endif  // JAMS_HAMILTONIAN_DIPOLE_INTERACTION_H
