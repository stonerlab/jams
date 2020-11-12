//
// Created by ioannis charalampidis on 07/11/2020.
//

#ifndef JAMS_SRC_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER_H_
#define JAMS_SRC_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER_H_

#include <jams/core/solver.h>
#include <jams/solvers/cpu_monte_carlo_metropolis.h>

#include <fstream>
#include <jams/core/types.h>
#include <pcg_random.hpp>
#include <random>
#include "jams/helpers/random.h"
#include <vector>
#include <ostream>
#include <iostream>

class MetadynamicsMetropolisSolver : public MetropolisMCSolver {
public:
    MetadynamicsMetropolisSolver() = default;

    ~MetadynamicsMetropolisSolver() override = default;

    void initialize(const libconfig::Setting &settings) override;

    void run() override;


    double energy_difference(const int spin_index,const Vec3 &initial_Spin,const Vec3 &final_Spin) override;

private:

    double potential;
    bool metadynamics;
//  auto potential_1d;

private:
    // I think its too expensive to do this in every iteration. Make a function to calculcate the mag depending if the
    // previous move has been accepted or not.
    static Vec3 total_magnetisation_calculation();

    static Vec3 trial_magnetisation_calculation(const Vec3 &current_magnetisation, const Vec3 &initial_spin, const Vec3 trial_spin);
    double potential_difference(const int spin_index,const Vec3 &initial_Spin,const Vec3 &final_Spin);

    std::vector<double> linear_space(const double &min, const double &max, const double &step);

    void intialise_potential_histograms();

    void insert_gaussian(const double &center, const double &amplitude,const double &width,const std::vector<double> &sample_points,std::vector<double> &discrete_potential);

    static double gaussian(const double &x, const double &center, const double &amplitude,const double &width);

    static double linear_interpolation(const double &x, const double &x_lower,const double &x_upper, const double &y_lower,const double &y_upper);

    double interpolated_potential(const std::vector<double> &sample_points,const std::vector<double> &discrete_potential,const double &value);

    // Monitor Function
	static double calculate_energy_difference(const std::vector<double> &potential); // energy barrier calculation for plain metadynamics


    const std::vector<double> sample_points_1d = linear_space(-2.0, 2.0,0.01); // mz with boundary conditions for the 1D potential
    const std::vector<double> sample_points_2d = linear_space(-1.0, 1.0,0.01); // Predefined m_z for 2D potential
    const std::vector<double> sample_points_m_perpendicular = linear_space(0, 1,0.01);//Predefined m_perp for 2D potential




    Vec3 magnetisation;
    Vec3 trial_magnetisation;
    std::vector<double> potential_1D;
    std::vector<std::vector<double>> potential_2D;
    const double gaussian_amplitude = 5.5e-24; // Height of the gaussian 1.0e-24 same as the paper
    const double gaussian_width = 0.03; // Width of the gaussian 1.4e-2 same as the paper

    std::ofstream energy_barrier_file;

};

#endif //JAMS_SRC_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER_H_
