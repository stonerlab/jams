// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CONSTRAINEDMC_H
#define JAMS_SOLVER_CONSTRAINEDMC_H

#include <fstream>

#include "core/solver.h"

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

class ConstrainedMCSolver : public Solver {
 public:
  ConstrainedMCSolver() : snew(0, 0), sigma(0, 0), eng(0, 0), move_acceptance_count_(0), move_acceptance_fraction_(1.0), move_sigma_(0.05) {}
  ~ConstrainedMCSolver();
  void initialize(int argc, char **argv, double dt);
  void run();
  bool is_converged();

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

  void AsselinAlgorithm(jblib::Vec3<double> (*mc_trial_step)(const jblib::Vec3<double>));
  void calculate_trial_move(jblib::Vec3<double> &spin, const double move_sigma);
  void set_spin(const int &i, const jblib::Vec3<double> &spin);
  void get_spin(const int &i, jblib::Vec3<double> &spin);
};

#endif  // JAMS_SOLVER_CONSTRAINEDMC_H
