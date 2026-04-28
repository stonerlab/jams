// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CONSTRAINEDMC_H
#define JAMS_SOLVER_CONSTRAINEDMC_H

#include <random>
#include <fstream>

#include <jams/core/types.h>
#include "jams/core/solver.h"
#include "jams/helpers/montecarlo.h"

#include "pcg_random.hpp"

class ConstrainedMCSolver : public Solver {
 public:
  ConstrainedMCSolver()  = default;
  ~ConstrainedMCSolver() = default;

  inline explicit ConstrainedMCSolver(const libconfig::Setting &settings) {
    initialize(settings);
  }

  void initialize(const libconfig::Setting& settings) override;
  void run() override;

  std::string name() const override { return "monte-carlo-constrained-cpu"; }

 private:
    void output_initialization_info(std::ostream &os);
    void output_running_stats_info(std::ostream &os);

    void validate_angles() const;
    void validate_rotation_matricies() const;
    void validate_moves() const;
    void validate_constraint() const;

    void sum_running_acceptance_statistics();
    void reset_running_statistics();

    void align_spins_to_constraint() const;

    unsigned AsselinAlgorithm(const std::function<jams::Vec<double, 3>(jams::Vec<double, 3>)>&  trial_spin_move);

    jams::Vec<double, 3>     rotate_cartesian_to_constraint(const int &i, const jams::Vec<double, 3> &spin) const;
    jams::Vec<double, 3>     rotate_constraint_to_cartesian(const int &i, const jams::Vec<double, 3> &spin) const;
    jams::Vec<double, 3>     total_transformed_magnetization() const;

    double   energy_difference(const int &s1, const jams::Vec<double, 3> &s1_initial, const jams::Vec<double, 3> &s1_trial, const int &s2, const jams::Vec<double, 3> &s2_initial, const jams::Vec<double, 3> &s2_trial) const;
    jams::Vec<double, 3>     magnetization_difference(const int &s1, const jams::Vec<double, 3> &s1_initial, const jams::Vec<double, 3> &s1_trial, const int &s2, const jams::Vec<double, 3> &s2_initial, const jams::Vec<double, 3> &s2_trial) const;

    bool do_spin_initial_alignment_ = true;

    double constraint_theta_   = 0.0;
    double constraint_phi_     = 0.0;
    jams::Vec<double, 3>   constraint_vector_  = {{0.0, 0.0, 1.0}};

    jams::Mat<double, 3, 3> rotation_matrix_         = kIdentityMat3;
    jams::Mat<double, 3, 3> inverse_rotation_matrix_ = kIdentityMat3;

    std::vector<jams::Mat<double, 3, 3>> spin_transformations_;

    int output_write_steps_ = 100;

    double move_fraction_uniform_     = 0.0;
    double move_fraction_angle_       = 1.0; // default is guaranteed erogodic
    double move_fraction_reflection_  = 0.0;

    double move_angle_sigma_ = 0.5;

    unsigned run_count_uniform_    = 0;
    unsigned run_count_angle_      = 0;
    unsigned run_count_reflection_ = 0;

    unsigned long long move_total_count_uniform_    = 0;
    unsigned long long move_total_count_angle_      = 0;
    unsigned long long move_total_count_reflection_ = 0;

    unsigned long long move_total_acceptance_count_uniform_      = 0;
    unsigned long long move_total_acceptance_count_angle_        = 0;
    unsigned long long move_total_acceptance_count_reflection_   = 0;

    unsigned long long move_running_acceptance_count_uniform_    = 0;
    unsigned long long move_running_acceptance_count_angle_      = 0;
    unsigned long long move_running_acceptance_count_reflection_ = 0;
};

#endif  // JAMS_SOLVER_CONSTRAINEDMC_H
