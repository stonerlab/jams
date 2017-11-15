// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CONSTRAINEDMC_H
#define JAMS_SOLVER_CONSTRAINEDMC_H

#include <random>
#include <fstream>

#include <jams/core/types.h>
#include "jams/core/solver.h"
#include "jams/core/montecarlo.h"

#include "pcg/pcg_random.hpp"

class ConstrainedMCSolver : public Solver {
 public:
  ConstrainedMCSolver()  = default;
  ~ConstrainedMCSolver() override = default;

  void initialize(const libconfig::Setting& settings) override;
  void run() override;
    bool is_running();

 private:
    unsigned AsselinAlgorithm(std::function<Vec3(Vec3)>  trial_spin_move);

    Vec3     rotate_cartesian_to_constraint(unsigned i, const Vec3 &spin) const;
    Vec3     rotate_constraint_to_cartesian(unsigned i, const Vec3 &spin) const;
    Vec3     total_transformed_magnetization() const;

    double   energy_difference(unsigned s1, const Vec3 &s1_initial, const Vec3 &s1_trial, unsigned s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const;
    Vec3     magnetization_difference(unsigned s1, const Vec3 &s1_initial, const Vec3 &s1_trial, unsigned s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const;

    void     validate_constraint() const;

    pcg64_k1024 random_generator_ = pcg_extras::seed_seq_from<std::random_device>();

    double constraint_theta_   = 0.0;
    double constraint_phi_     = 0.0;
    Vec3   constraint_vector_  = {0.0, 0.0, 1.0};

    Mat3 rotation_matrix_         = kIdentityMat3;
    Mat3 inverse_rotation_matrix_ = kIdentityMat3;

    std::vector<Mat3> spin_transformations_;

    int max_run_steps_ = 1;

    int output_write_steps_ = 100;

    double move_fraction_uniform_     = 1.0; // default is guaranteed erogodic
    double move_fraction_angle_       = 0.0;
    double move_fraction_reflection_  = 0.0;

    double move_angle_sigma_ = 0.5;

    unsigned run_count_uniform    = 0;
    unsigned run_count_angle      = 0;
    unsigned run_count_reflection = 0;

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
