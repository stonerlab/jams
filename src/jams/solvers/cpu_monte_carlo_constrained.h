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
  ~ConstrainedMCSolver() override = default;

  void initialize(const libconfig::Setting& settings) override;
  void run() override;

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

    unsigned AsselinAlgorithm(const std::function<Vec3(Vec3)>&  trial_spin_move);

    Vec3     rotate_cartesian_to_constraint(const int &i, const Vec3 &spin) const;
    Vec3     rotate_constraint_to_cartesian(const int &i, const Vec3 &spin) const;
    Vec3     total_transformed_magnetization() const;

    double   energy_difference(const int &s1, const Vec3 &s1_initial, const Vec3 &s1_trial, const int &s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const;
    Vec3     magnetization_difference(const int &s1, const Vec3 &s1_initial, const Vec3 &s1_trial, const int &s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const;

    bool do_spin_initial_alignment_ = true;

    double constraint_theta_   = 0.0;
    double constraint_phi_     = 0.0;
    Vec3   constraint_vector_  = {{0.0, 0.0, 1.0}};

    Mat3 rotation_matrix_         = kIdentityMat3;
    Mat3 inverse_rotation_matrix_ = kIdentityMat3;

    std::vector<Mat3> spin_transformations_;

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
