#ifndef JAMS_NULL_SOLVER_H
#define JAMS_NULL_SOLVER_H

#include <jams/core/solver.h>

//
// This solver does almost nothing. It initializes the base class and performs
// a single iteration (which has no effect). It allows all of the mechanics
// of initialising a simulation to be run without choosing a specific solver.
// It is particularly useful for running monitors without doing any solving
// but setting "output_steps=1" for the monitor.
//
class NullSolver : public Solver {
public:
    NullSolver() = default;
    ~NullSolver() override = default;

    inline explicit NullSolver(const libconfig::Setting& settings) {
      initialize(settings);
    }

    std::string name() const override { return "null"; }


    inline void initialize(const libconfig::Setting& settings) final {
      max_steps_ = 1;
    }

    inline void run() final {
      iteration_++;
      time_ = iteration_ * step_size_;
    }
};


#endif //JAMS_NULL_SOLVER_H
