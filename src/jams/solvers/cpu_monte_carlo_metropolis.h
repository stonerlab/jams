// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_METROPOLISMC_H
#define JAMS_SOLVER_METROPOLISMC_H

#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <jams/core/solver.h>
#include <jams/interface/config.h>

///
/// Implements a solver based on Metropolis Monte Carlo. The class adds further
/// virtual members so that it can be derived again for solvers which use
/// Metropolis Monte Carlo as a template algorithm.
///

class MetropolisMCSolver : public Solver {
public:
    using MoveFunction = std::function<Vec3(Vec3)>;

    MetropolisMCSolver() = default;

    ~MetropolisMCSolver() override = default;

    void initialize(const libconfig::Setting &settings) override;

    void run() override;

    /// Calculates the energy difference of the system when the spin at
    /// @p spin_index is set to @p initial_spin and @p final_spin.
    /// \f[
    /// \Delta E = E_{final} - E_{initial}
    /// \f]
    /// A negative energy difference means the final state is lower in energy
    /// than the initial state.
    virtual double energy_difference(const int spin_index,
                                     const Vec3 &initial_spin,
                                     const Vec3 &final_spin);

    /// Performs 'one' Monte Carlo step. We define as one attempted move of
    /// every spin in the system on average. This means we don't guarantee
    /// every spin will be touched each step, but there is a uniform
    /// probability. Hence one step involves num_spins trial moves.
    virtual int monte_carlo_step(const MoveFunction &trial_spin_move);

    /// Implements the Metropolis algorithm where trial spin moves are accepted
    /// or rejected according to the Boltzmann distribution. @p trial_spin_move
    /// accepts any function with the signature MoveFunction allowing the
    /// algorithm to be used with different trial moves. Some standard moves are
    /// provided in src/jams/montecarlo.h as functor classes which implement
    /// the MonteCarloMove abstract class.
    virtual int
    metropolis_algorithm(const MoveFunction &trial_spin_move, int spin_index);

private:
    /// Outputs statistics of Monte Carlo move acceptance rates to a file.
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
