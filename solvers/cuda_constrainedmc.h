// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDACONSTRAINEDMC_H
#define JAMS_SOLVER_CUDACONSTRAINEDMC_H

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

  curandGenerator_t gen; // device random generator
  jblib::Array<double, 1> random_number_buffer_;
  jblib::CudaArray<double, 1> dev_random_number_buffer_;
  int curand_iterator_;

  double get_uniform_random_number();
  double compute_one_spin_energy(const jblib::Vec3<double> &s_final, const int &ii);
  void calculate_trial_move(jblib::Vec3<double> &spin);
};

#endif  // JAMS_SOLVER_CUDACONSTRAINEDMC_H
