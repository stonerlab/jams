#ifndef JAMS_HAMILTONIAN_ENERGY_CURRENT_INTERACTION_H
#define JAMS_HAMILTONIAN_ENERGY_CURRENT_INTERACTION_H

#include <jams/containers/mat3.h>
#include <jams/containers/vec3.h>

namespace jams {

class EnergyCurrentInteractionSink {
public:
  virtual ~EnergyCurrentInteractionSink() = default;

  // r_ji is the current-formula displacement r_j - r_i.
  virtual void insert(int site_i,
                      int site_j,
                      const jams::Vec<double, 3>& r_ji,
                      const jams::Mat<double, 3, 3>& interaction) = 0;
};

}  // namespace jams

#endif  // JAMS_HAMILTONIAN_ENERGY_CURRENT_INTERACTION_H
