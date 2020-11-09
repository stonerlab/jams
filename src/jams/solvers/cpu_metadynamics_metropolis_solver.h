//
// Created by ioannis charalampidis on 07/11/2020.
//

#ifndef JAMS_SRC_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER_H_
#define JAMS_SRC_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER_H_

#include "jams/core/solver.h"

#include <fstream>
#include <jams/core/types.h>
#include <pcg_random.hpp>
#include <random>
#include "jams/helpers/random.h"
#include <vector>


class MetadynamicsMetropolisSolver {
 public:

  double potential;
  double s_initial;
  double s_final;
  bool metadynamics;
  auto potential_1d;

 private:
  std::vector<double> linear_space(const double& min, const double& max, const double& step);
  const double gaussian_amplitude = 1e-24; // Height of the gaussian 1.0e-24 same as the paper
  const double gaussian_width = 0.03; // Width of the gaussian 1.4e-2 same as the paper
  //want to use the sizes of these vector to initialise the potential histograms
  const std::vector<double> sample_points_1d = linear_space(-2.0, 2.0, 0.01); // mz with boundary conditions for the 1D potential
  const std::vector<double> sample_points_2d = linear_space(-1.0, 1.0, 0.01); // Predefined m_z for 2D potential
  const std::vector<double> sample_points_m_perpendicular = linear_space (0,1,0.01);//Predefined m_perp for 2D potential

 // std::vector<double> potential_1D(sample_points_1d.size(), 0.0);
//  std::vector<std::vector<double>> potential_2D(sample_points_2d.size(), std::vector<double>(sample_points_m_perpendicular.size(), 0.0));

 protected:


};

#endif //JAMS_SRC_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER_H_
