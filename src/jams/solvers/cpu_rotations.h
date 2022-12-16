//
// Created by Joseph Barker on 2019-05-03.
//

#ifndef JAMS_SOLVERS_CPU_ROTATIONS_H
#define JAMS_SOLVERS_CPU_ROTATIONS_H

#include "jams/core/solver.h"

class RotationSolver : public Solver {
public:
    RotationSolver() = default;
    ~RotationSolver() override = default;

    inline explicit RotationSolver(const libconfig::Setting &settings) {
      initialize(settings);
    }

    bool is_running() override {
      return iteration_ == 0;
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "rotations-cpu"; }


private:
    unsigned num_theta_ = 36;
    unsigned num_phi_   = 72;
};

#endif //JAMS_SOLVERS_CPU_ROTATIONS_H
