// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CONSTRAINEDMC_H
#define JAMS_SOLVER_CONSTRAINEDMC_H

#include <random>
#include <fstream>
#include <jams/core/types.h>

#include "jams/core/solver.h"
#include "jams/core/montecarlo.h"

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

#include "pcg/pcg_random.hpp"

class ConstrainedMCSolver : public Solver {
 public:
  ConstrainedMCSolver() :
          snew(0, 0),
          sigma(0, 0),
          eng(0, 0),
          move_acceptance_fraction_(1.0),
    random_generator_(pcg_extras::seed_seq_from<std::random_device>()),
    random_uniform_int_distribution_(0, globals::num_spins-1),
    random_uniform_real_distribution_(0, 1.0)

    {}
  ~ConstrainedMCSolver();
  void initialize(int argc, char **argv, double dt);
  void run();
  bool is_converged();

 private:
    unsigned AsselinAlgorithm(std::function<Vec3(Vec3)>  move);
    void calculate_trial_move(Vec3 &spin, const double move_sigma);
    void set_spin(const int &i, const Vec3 &spin);
    void get_spin(const int &i, Vec3 &spin);

    void configure_move_types(const libconfig::Setting& config);

    unsigned output_write_steps = 100;

    double move_fraction_uniform_     = 1.0; // default is guaranteed erogodic
    double move_fraction_angle_       = 0.0;
    double move_fraction_reflection_  = 0.0;

    double move_angle_sigma_ = 0.1;

    unsigned run_count_uniform    = 0;
    unsigned run_count_angle      = 0;
    unsigned run_count_reflection = 0;

    unsigned long long move_total_count_uniform_ = 0;
    unsigned long long move_total_count_angle_ = 0;
    unsigned long long move_total_count_reflection_ = 0;

    unsigned long long move_total_acceptance_count_uniform_ = 0;
    unsigned long long move_total_acceptance_count_angle_ = 0;
    unsigned long long move_total_acceptance_count_reflection_ = 0;

    unsigned long long move_running_acceptance_count_uniform_ = 0;
    unsigned long long move_running_acceptance_count_angle_ = 0;
    unsigned long long move_running_acceptance_count_reflection_ = 0;

    double move_acceptance_fraction_ = 0.0;

    double constraint_theta_ = 0.0;
    double constraint_phi_ = 0.0;

    jblib::Array<double, 2> snew;
    jblib::Array<double, 2> sigma;
    jblib::Array<double, 2> eng;
    jblib::Array<Mat3, 1> s_transform_;


    pcg32 random_generator_;
    std::uniform_int_distribution<> random_uniform_int_distribution_;
    std::uniform_real_distribution<> random_uniform_real_distribution_;

  Vec3 constraint_vector_;
  Mat3 rotation_matrix_;
  Mat3 inverse_rotation_matrix_;
  std::ofstream outfile;


};

#endif  // JAMS_SOLVER_CONSTRAINEDMC_H
