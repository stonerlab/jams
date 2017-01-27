// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_MONTE_CARLO_WOLFF_H
#define JAMS_SOLVER_MONTE_CARLO_WOLFF_H

#include "jams/core/solver.h"

#include <queue>
#include <vector>

typedef std::vector<std::vector<int>> adjacency_list;

class MonteCarloWolffSolver : public Solver {
 public:
  MonteCarloWolffSolver();
  ~MonteCarloWolffSolver();

  void initialize(int argc, char **argv, double dt);
  void run();
  int cluster_move();

 private:
   double r_cutoff_;
   int attempted_moves_;
   adjacency_list    neighbours_;
};

#endif  // JAMS_SOLVER_MONTE_CARLO_WOLFF_H
