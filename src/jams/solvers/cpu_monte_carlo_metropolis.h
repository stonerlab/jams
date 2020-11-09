// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_METROPOLISMC_H
#define JAMS_SOLVER_METROPOLISMC_H

#include "jams/core/solver.h"

#include <fstream>
#include <jams/core/types.h>
#include <pcg_random.hpp>
#include <random>
#include "jams/helpers/random.h"
#include <jams/helpers/montecarlo.h>

class MetropolisMCSolver : public Solver {
public:
    using MoveFunction = std::function<Vec3(Vec3)>;

    MetropolisMCSolver() = default;

    ~MetropolisMCSolver() override = default;

    void initialize(const libconfig::Setting &settings) override;

    void run() override;

    virtual double energy_difference(const int spin_index,
                                     const Vec3 &initial_spin,
                                     const Vec3 &final_spin);

    virtual int monte_carlo_step(const MoveFunction &trial_spin_move);

    virtual int
    metropolis_algorithm(const MoveFunction &trial_spin_move, int spin_index);

private:
    void output_move_statistics();

    std::ofstream stats_file_;

    std::vector<std::string> move_names_;
    std::vector<double> move_weights_;
    std::vector<MoveFunction> move_functions_;
    std::vector<int> moves_attempted_;
    std::vector<int> moves_accepted_;

    int output_write_steps_ = 1000;
};

#endif  // JAMS_SOLVER_METROPOLISMC_H
