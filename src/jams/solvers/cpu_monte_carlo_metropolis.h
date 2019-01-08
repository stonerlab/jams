// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_METROPOLISMC_H
#define JAMS_SOLVER_METROPOLISMC_H

#include "jams/core/solver.h"

#include "jblib/containers/vec.h"
#include "jblib/containers/array.h"

#include <fstream>
#include <jams/core/types.h>
#include <pcg/pcg_random.hpp>
#include <random>
#include "jams/helpers/random.h"

class MetropolisMCSolver : public Solver {
 public:
  MetropolisMCSolver() = default;
  ~MetropolisMCSolver() = default;
  void initialize(const libconfig::Setting& settings);
  void run();

 private:
  class MagnetizationRotationMinimizer;

  int MetropolisAlgorithm(std::function<Vec3(Vec3)> trial_spin_move);
    int MetropolisAlgorithmTotalEnergy(std::function<Vec3(Vec3)> trial_spin_move);
    void MetropolisPreconditioner(std::function<Vec3(Vec3)>  trial_spin_move);
  void SystematicPreconditioner(const double delta_theta, const double delta_phi);

  pcg32_k1024 random_generator_ = pcg_extras::seed_seq_from<pcg32>(jams::random_generator());

  bool use_total_energy_ = false;
  bool is_preconditioner_enabled_ = false;
  double preconditioner_delta_theta_ = 5.0;
  double preconditioner_delta_phi_ = 5.0;

  int output_write_steps_ = 1000;

  double move_fraction_uniform_     = 0.4;
  double move_fraction_angle_       = 0.6;
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

#endif  // JAMS_SOLVER_METROPOLISMC_H
