// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_HEUNLLG_H
#define JAMS_SOLVER_HEUNLLG_H

#include <random>
#include <libconfig.h++>

#include <pcg_random.hpp>

#include "jams/core/solver.h"
#include "jams/helpers/random.h"
#include "jams/containers/multiarray.h"

class HeunLLGSolver : public Solver {
public:
  HeunLLGSolver() = default;
  ~HeunLLGSolver() override = default;

  inline explicit HeunLLGSolver(const libconfig::Setting &settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override;
  void run() override;

  std::string name() const override { return "llg-heun-cpu"; }

 private:
    jams::MultiArray<double, 2> s_old_;
    jams::MultiArray<double, 1> sigma_;
    jams::MultiArray<double, 2> w_;

    pcg32_k1024 random_generator_ = pcg_extras::seed_seq_from<pcg32>(jams::instance().random_generator()());
};

#endif  // JAMS_SOLVER_HEUNLLG_H
