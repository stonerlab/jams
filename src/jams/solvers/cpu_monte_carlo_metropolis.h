// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_METROPOLISMC_H
#define JAMS_SOLVER_METROPOLISMC_H

#include "jams/core/solver.h"

#include <fstream>
#include <jams/core/types.h>
#include <pcg_random.hpp>
#include <random>
#include "jams/helpers/random.h"

class MetropolisMCSolver : public Solver {
 public:
    using SpinMoveFunction = std::function<Vec3(Vec3)>;

  MetropolisMCSolver() = default;
  ~MetropolisMCSolver() = default;
  void initialize(const libconfig::Setting& settings);
  void run();

 private:
  class MagnetizationRotationMinimizer;
  int MetropolisStep(SpinMoveFunction trial_spin_move);
  double Metropolis_Energy_Difference(const Vec3 &initial_Spin, const Vec3 &final_Spin,const int &spin_index);
  int MetropolisAlgorithm(SpinMoveFunction trial_spin_move,int &spin_index);
  bool acceptance_with_boltzmann_distribution(const double &deltaE, const double &beta);
    int MetropolisAlgorithmTotalEnergy(SpinMoveFunction trial_spin_move);
    void MetropolisPreconditioner(SpinMoveFunction  trial_spin_move);
  void SystematicPreconditioner(const double delta_theta, const double delta_phi);

  bool metadynamics_applied = false;
  bool use_random_spin_order_ = true;
  bool use_total_energy_ = false;
  bool is_preconditioner_enabled_ = false;
  double preconditioner_delta_theta_ = 5.0;
  double preconditioner_delta_phi_ = 5.0;

  int output_write_steps_ = 1000;

  double move_fraction_uniform_     = 0.0;
  double move_fraction_angle_       = 1.0;
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

#endif  // JAMS_SOLVER_METROPOLISMC_H
