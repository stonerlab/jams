// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_HEUNLLG_H
#define JAMS_SOLVER_HEUNLLG_H

#include <libconfig.h++>

#include "jams/core/solver.h"

#include "jblib/containers/array.h"

class HeunLLGSolver : public Solver {
 public:
  HeunLLGSolver() : snew(0, 0), sigma(0), eng(0, 0), w(0, 0) {}
  ~HeunLLGSolver() {}
  void initialize(const libconfig::Setting& settings);
  void run();


 private:
    double dt;

    jblib::Array<double, 2> snew;
    jblib::Array<double, 1> sigma;
    jblib::Array<double, 2> eng;
    jblib::Array<double, 2> w;
};

#endif  // JAMS_SOLVER_HEUNLLG_H
