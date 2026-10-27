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

  bool supports_deterministic_spin_rate() const override { return true; }
  double max_deterministic_spin_rate() override {
    return max_llg_spin_rate_cpu(gyro_eff_);
  }

 private:
    jams::MultiArray<double, 2> s_old_;
    jams::MultiArray<jams::Real, 1> gyro_eff_;
};

#endif  // JAMS_SOLVER_HEUNLLG_H
