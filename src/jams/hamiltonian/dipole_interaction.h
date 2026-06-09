#ifndef JAMS_HAMILTONIAN_DIPOLE_INTERACTION_H
#define JAMS_HAMILTONIAN_DIPOLE_INTERACTION_H

#include <cmath>
#include <stdexcept>

#include <jams/containers/mat3.h>
#include <jams/containers/vec3.h>
#include <jams/hamiltonian/energy_current_interaction.h>
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
    jams::EnergyCurrentInteractionSink& sink,
    const int site_i,
    const int site_j,
    const jams::Vec<double, 3>& r_ji,
    const jams::Mat<double, 3, 3>& interaction) {
  sink.insert(site_i, site_j, r_ji, interaction);
}

}  // namespace jams::dipole

#endif  // JAMS_HAMILTONIAN_DIPOLE_INTERACTION_H
