// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CONSTRAINEDMC_H
#define JAMS_SOLVER_CONSTRAINEDMC_H

#include "core/solver.h"

#include "jblib/containers/array.h"

class ConstrainedMCSolver : public Solver {
 public:
  ConstrainedMCSolver() : snew(0, 0), sigma(0, 0), eng(0, 0), move_acceptance_count_(0), move_acceptance_fraction_(1.0), move_sigma_(0.05) {}
  ~ConstrainedMCSolver() {}
  void initialize(int argc, char **argv, double dt);
  void run();
  void compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t,
    double &e4_s);

 private:
  jblib::Array<double, 2> snew;
  jblib::Array<double, 2> sigma;
  jblib::Array<double, 2> eng;

  int    move_acceptance_count_;
  double move_acceptance_fraction_;
  double move_sigma_;

  double constraint_theta_;
  double constraint_phi_;
  jblib::Vec3<double> constraint_vector_;
  jblib::Matrix<double, 3, 3> rotation_matrix_;
  jblib::Matrix<double, 3, 3> inverse_rotation_matrix_;
  std::ofstream outfile;


  double compute_one_spin_energy(const jblib::Vec3<double> &s_final, const int &ii);
  void calculate_trial_move(jblib::Vec3<double> &spin, const double move_sigma);
};

#endif  // JAMS_SOLVER_CONSTRAINEDMC_H
