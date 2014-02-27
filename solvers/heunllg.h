// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_HEUNLLG_H
#define JAMS_SOLVER_HEUNLLG_H

#include "core/solver.h"

#include "jblib/containers/array.h"

class HeunLLGSolver : public Solver {
 public:
  HeunLLGSolver() : snew(0, 0), sigma(0), eng(0, 0), w(0, 0) {}
  ~HeunLLGSolver() {}
  void initialize(int argc, char **argv, double dt);
  void run();
  void compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t,
    double &e4_s);

 private:
  jblib::Array<double, 2> snew;
  jblib::Array<double, 1> sigma;
  jblib::Array<double, 2> eng;
  jblib::Array<double, 2> w;
};

#endif  // JAMS_SOLVER_HEUNLLG_H