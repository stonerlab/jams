// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDACONSTRAINEDMC_H
#define JAMS_SOLVER_CUDACONSTRAINEDMC_H

#ifdef CUDA

#include <fstream>

#include <curand.h>

#include "jams/core/cuda_solver.h"

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"
#include "jblib/containers/vec.h"

class CudaConstrainedMCSolver : public CudaSolver {
 public:
  CudaConstrainedMCSolver() : snew(0, 0), sigma(0, 0), eng(0, 0), move_acceptance_count_(0), move_acceptance_fraction_(1.0), move_sigma_(0.05) {}
  ~CudaConstrainedMCSolver() {}
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
  Vec3 constraint_vector_;
  Mat3 rotation_matrix_;
  Mat3 inverse_rotation_matrix_;
  std::ofstream outfile;

  double compute_one_spin_energy(const Vec3 &s_final, const int &ii);
  void calculate_trial_move(Vec3 &spin, const double move_sigma);
  void set_spin(const int &i, Vec3 &spin);
  void get_spin(const int &i, Vec3 &spin);
};

#endif  // CUDA

#endif  // JAMS_SOLVER_CUDACONSTRAINEDMC_H
