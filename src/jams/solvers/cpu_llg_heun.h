// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_HEUNLLG_H
#define JAMS_SOLVER_HEUNLLG_H

#include <random>
#include <libconfig.h++>

#include <pcg/pcg_random.hpp>

#include "jams/core/solver.h"
#include "jams/helpers/random.h"
#include "jblib/containers/array.h"

class HeunLLGSolver : public Solver {
 public:
  HeunLLGSolver() = default;
  ~HeunLLGSolver() = default;
  void initialize(const libconfig::Setting& settings);
  void run();


 private:
    double dt = 0.0;

    jblib::Array<double, 2> snew;
    jblib::Array<double, 1> sigma;
    jblib::Array<double, 2> w;

    pcg32_k1024 random_generator_ = pcg_extras::seed_seq_from<pcg32>(jams::random_generator());
};

#endif  // JAMS_SOLVER_HEUNLLG_H
