// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_METROPOLISMC_H
#define JAMS_SOLVER_METROPOLISMC_H

#include "core/solver.h"

#include "jblib/containers/array.h"

class MetropolisMCSolver : public Solver {
 public:
  MetropolisMCSolver() : snew(0, 0), sigma(0, 0), eng(0, 0) {}
  ~MetropolisMCSolver() {}
  void initialize(int argc, char **argv, double dt);
  void run();
  void compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t,
    double &e4_s);

 private:
  jblib::Array<double, 2> snew;
  jblib::Array<double, 2> sigma;
  jblib::Array<double, 2> eng;

  void oneSpinEnergy(const int &i, double total[3]);
};

#endif  // JAMS_SOLVER_METROPOLISMC_H
