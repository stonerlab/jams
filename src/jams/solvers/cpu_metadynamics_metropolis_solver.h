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

#include <jams/metadynamics/magnetisation_cv.h>

class MetadynamicsMetropolisSolver : public MetropolisMCSolver {
public:
    MetadynamicsMetropolisSolver() = default;

    ~MetadynamicsMetropolisSolver() override = default;

    void initialize(const libconfig::Setting &settings) override;

    void run() override;

    double energy_difference(const int spin_index, const Vec3 &initial_Spin,
                             const Vec3 &final_Spin) override;

    void accept_move(const int spin_index, const Vec3 &initial_spin,
                     const Vec3 &final_spin) override;

private:

    double tempering_amplitude();

    std::ofstream metadynamics_stats;

    std::unique_ptr<jams::CollectiveVariablePotential> cv_potential_;
    int gaussian_deposition_stride_;

    bool do_tempering_ = false;
    double tempering_bias_temperature_;
};

#endif //JAMS_SRC_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER_H_
