#ifndef JAMS_INITIALIZER_INIT_SPIN_PERTURBATION_H
#define JAMS_INITIALIZER_INIT_SPIN_PERTURBATION_H

#include <jams/interface/config.h>

namespace jams {

class InitSpinPerturbation {
 public:
  static void execute(const libconfig::Setting& settings);
};

}  // namespace jams
#endif
