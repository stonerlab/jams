// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_GLOBALS_H
#define JAMS_CORE_GLOBALS_H

#include <string>

#include "jblib/containers/array.h"

#ifndef GLOBALORIGIN
#define GLOBAL extern
#else
#define GLOBAL
#endif

class Solver;
class Lattice;
class Random;
namespace libconfig { class Config; }

GLOBAL Solver  *solver;
GLOBAL Lattice *lattice;
GLOBAL libconfig::Config *config;  ///< Config object
GLOBAL Random  *rng;
GLOBAL std::string seedname;

namespace globals {
  GLOBAL unsigned int num_spins;
  GLOBAL unsigned int num_spins3;

  GLOBAL jblib::Array<double, 2> s;
  GLOBAL jblib::Array<double, 2> h;
  GLOBAL jblib::Array<double, 2> ds_dt;

  GLOBAL jblib::Array<double, 1> alpha;
  GLOBAL jblib::Array<double, 1> mus;
  GLOBAL jblib::Array<double, 1> gyro;
}  // namespace globals
#undef GLOBAL
#endif  // JAMS_CORE_GLOBALS_H
