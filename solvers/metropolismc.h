// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_METROPOLISMC_H
#define JAMS_SOLVER_METROPOLISMC_H

#include "core/solver.h"

#include "jblib/containers/vec.h"
#include "jblib/containers/array.h"

class MetropolisMCSolver : public Solver {
 public:
  MetropolisMCSolver() : snew(0, 0), sigma(0, 0), eng(0, 0) {}
  ~MetropolisMCSolver() {}
  void initialize(int argc, char **argv, double dt);
  void run();

 private:

  void MetropolisAlgorithm(jblib::Vec3<double> (*mc_move)(const jblib::Vec3<double>));

  jblib::Array<double, 2> snew;
  jblib::Array<double, 2> sigma;
  jblib::Array<double, 2> eng;
};

#endif  // JAMS_SOLVER_METROPOLISMC_H
