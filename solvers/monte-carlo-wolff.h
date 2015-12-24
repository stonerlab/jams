// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_MONTE_CARLO_WOLFF_H
#define JAMS_SOLVER_MONTE_CARLO_WOLFF_H

#include "core/solver.h"

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
   int attempted_moves_;
   int cluster_size_;
   adjacency_list    neighbours_;
};

#endif  // JAMS_SOLVER_MONTE_CARLO_WOLFF_H
