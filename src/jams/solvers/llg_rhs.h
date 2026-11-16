// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVERS_LLG_RHS_H
#define JAMS_SOLVERS_LLG_RHS_H

#include "jams/helpers/finite.h"

namespace jams::solvers {

// field includes the thermal field, in Tesla. The result is in ps^-1.
JAMS_HOST_DEVICE inline void llg_rhs(const double spin[3], const double field[3],
                                     const double gyro_eff, const double alpha, double slope[3]) {
  const double cx = spin[1] * field[2] - spin[2] * field[1];
  const double cy = spin[2] * field[0] - spin[0] * field[2];
  const double cz = spin[0] * field[1] - spin[1] * field[0];

  const double scx = spin[1] * cz - spin[2] * cy;
  const double scy = spin[2] * cx - spin[0] * cz;
  const double scz = spin[0] * cy - spin[1] * cx;

  slope[0] = -gyro_eff * (cx + alpha * scx);
  slope[1] = -gyro_eff * (cy + alpha * scy);
  slope[2] = -gyro_eff * (cz + alpha * scz);
}

}  // namespace jams::solvers

#endif  // JAMS_SOLVERS_LLG_RHS_H
