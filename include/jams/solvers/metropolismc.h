// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_METROPOLISMC_H
#define JAMS_SOLVER_METROPOLISMC_H

#include "jams/core/solver.h"

#include "jblib/containers/vec.h"
#include "jblib/containers/array.h"

#include <fstream>
#include <jams/core/types.h>

class MetropolisMCSolver : public Solver {
 public:
  MetropolisMCSolver() : snew(0, 0), sigma(0, 0), eng(0, 0) {}
  ~MetropolisMCSolver();
  void initialize(int argc, char **argv, double dt);
  void run();

 private:

  class MagnetizationRotationMinimizer;

  void MetropolisAlgorithm(Vec3 (*mc_move)(const Vec3));
  void MetropolisPreconditioner(Vec3 (*mc_trial_step)(const Vec3));
  void SystematicPreconditioner(const double delta_theta, const double delta_phi);

  jblib::Array<double, 2> snew;
  jblib::Array<double, 2> sigma;
  jblib::Array<double, 2> eng;

  bool is_preconditioner_enabled_;
  double preconditioner_delta_theta_;
  double preconditioner_delta_phi_;

  int    move_acceptance_count_;
  double move_acceptance_fraction_;
  std::ofstream outfile;
};

#endif  // JAMS_SOLVER_METROPOLISMC_H
