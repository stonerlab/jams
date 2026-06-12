// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVERS_CPU_LLG_RKMK_H
#define JAMS_SOLVERS_CPU_LLG_RKMK_H

#include <libconfig.h++>

#include "jams/containers/multiarray.h"
#include "jams/core/solver.h"

class RKMK2LLGSolver : public Solver {
 public:
  RKMK2LLGSolver() = default;
  ~RKMK2LLGSolver() override = default;

  inline explicit RKMK2LLGSolver(const libconfig::Setting& settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override;
  void run() override;

  std::string name() const override { return "llg-rkmk2-cpu"; }

 private:
  jams::MultiArray<jams::Real, 1> gyro_eff_;
  jams::MultiArray<double, 2> s_init_;
  jams::MultiArray<double, 2> phi_;
};

class RKMK4LLGSolver : public Solver {
 public:
  RKMK4LLGSolver() = default;
  ~RKMK4LLGSolver() override = default;

  inline explicit RKMK4LLGSolver(const libconfig::Setting& settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override;
  void run() override;

  std::string name() const override { return "llg-rkmk4-cpu"; }

 private:
  jams::MultiArray<jams::Real, 1> gyro_eff_;
  jams::MultiArray<double, 2> s_init_;
  jams::MultiArray<double, 2> k1_;
  jams::MultiArray<double, 2> k2_;
  jams::MultiArray<double, 2> k3_;
};

#endif  // JAMS_SOLVERS_CPU_LLG_RKMK_H
