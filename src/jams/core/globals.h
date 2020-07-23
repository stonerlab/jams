// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_GLOBALS_H
#define JAMS_CORE_GLOBALS_H

#include <string>

#include "jams/containers/multiarray.h"

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif

class Solver;
class Lattice;
namespace libconfig { class Config; }

GLOBAL Solver  *solver;
GLOBAL Lattice *lattice;
GLOBAL std::unique_ptr<libconfig::Config> config;  ///< Config object
GLOBAL std::string simulation_name;

namespace globals {
  GLOBAL int num_spins;
  GLOBAL int num_spins3;

  GLOBAL jams::MultiArray<double, 2> s;
  GLOBAL jams::MultiArray<double, 2> h;
  GLOBAL jams::MultiArray<double, 2> ds_dt;

  GLOBAL jams::MultiArray<double, 1> alpha;
  GLOBAL jams::MultiArray<double, 1> mus;
  GLOBAL jams::MultiArray<double, 1> gyro;
}  // namespace globals
#undef GLOBAL
#endif  // JAMS_CORE_GLOBALS_H
