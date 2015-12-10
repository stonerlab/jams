// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_MONTE_CARLO_WOLFF_H
#define JAMS_SOLVER_MONTE_CARLO_WOLFF_H

#include "core/solver.h"

class MonteCarloWolffSolver : public Solver {
 public:
  MonteCarloWolffSolver();
  ~MonteCarloWolffSolver();
  void initialize(int argc, char **argv, double dt);
  void run();

 private:
};

#endif  // JAMS_SOLVER_MONTE_CARLO_WOLFF_H
