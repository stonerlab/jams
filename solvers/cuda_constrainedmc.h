// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDACONSTRAINEDMC_H
#define JAMS_SOLVER_CUDACONSTRAINEDMC_H

#ifdef CUDA

#include <curand.h>

#include "core/cuda_solver.h"

#include "jblib/containers/array.h"

class CudaConstrainedMCSolver : public CudaSolver {
 public:
  CudaConstrainedMCSolver() : snew(0, 0), sigma(0, 0), eng(0, 0) {}
  ~CudaConstrainedMCSolver() {}
  void initialize(int argc, char **argv, double dt);
  void run();
  void compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t,
    double &e4_s);

 private:
  jblib::Array<double, 2> snew;
  jblib::Array<double, 2> sigma;
  jblib::Array<double, 2> eng;

  double constraint_theta_;
  double constraint_phi_;
  jblib::Vec3<double> constraint_vector_;
  jblib::Matrix<double, 3, 3> rotation_matrix_;
  jblib::Matrix<double, 3, 3> inverse_rotation_matrix_;

  double compute_one_spin_energy(const jblib::Vec3<double> &s_final, const int &ii);
  void calculate_trial_move(jblib::Vec3<double> &spin);
};

#endif  // CUDA

#endif  // JAMS_SOLVER_CUDACONSTRAINEDMC_H