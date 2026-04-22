// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_HEUNLLG_H
#define JAMS_SOLVER_HEUNLLG_H

#include <libconfig.h++>

#include <jams/common.h>
#include "jams/core/solver.h"
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
    jams::MultiArray<double, 2> extra_torque_;
};

#endif  // JAMS_SOLVER_HEUNLLG_H
