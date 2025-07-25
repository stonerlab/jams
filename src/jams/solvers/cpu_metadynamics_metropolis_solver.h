// cpu_metadynamics_metropolis_solver.h                                -*-C++-*-

#ifndef INCLUDED_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER
#define INCLUDED_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER

#include "jams/solvers/cpu_monte_carlo_metropolis.h"

#include "jams/metadynamics/metadynamics_potential.h"

#include <fstream>
#include <memory>
#include <vector>

// ==================================
// class MetadynamicsMetropolisSolver
// ==================================

class MetadynamicsMetropolisSolver : public MetropolisMCSolver {
private:
    /// Pointer to collective variable object (owned by this class)
    std::unique_ptr<jams::MetadynamicsPotential> metad_potential_;

    /// Number of solver iterations between gaussian depositions into the
    /// collective variable potential landscape.
    int gaussian_deposition_stride_;

    /// Number of iterations to run before we start inserting gaussians
    int gaussian_deposition_delay_;

    /// Number of solver iterations between outputing the potential.
    int output_steps_;

    /// Toggle whether tempered metadynamics (systematically reducing the
    /// gaussian amplitude) is used.
    bool do_tempering_ = false;

    /// Bias temperature for tempered metadynamics algorithm.
    double tempering_bias_temperature_;

    /// File name for the shared potential file for parallel calculations
    std::string parallel_shared_potential_file_name_;

    /// Number of steps between synchronising the potential with the shared file for parallel calculations
    int parallel_shared_potential_sync_steps_;

    /// Output file for metadynamics statistics
    std::ofstream metadynamics_stats_file_;

public:
    /// CREATORS

    /// Default constructor
    MetadynamicsMetropolisSolver() = default;

    /// Default destructor
    ~MetadynamicsMetropolisSolver() override = default;

    inline explicit MetadynamicsMetropolisSolver(const libconfig::Setting &settings) {
      initialize(settings);
    }

    // Initializes the MetadynamicsMetropolisSolver using settings from the global
    // config. This will also create and attach a coordinate variable object
    // specified in the config.
    void initialize(const libconfig::Setting &settings) override;

    // Runs the metadynamics solver. This involves running the underlying Metropolis
    // algorithm and also triggering the insertion of gaussians into the potential
    // energy landscape.
    void run() override;

    // Overrides the energy_difference function of MetropolisMCSolver
    // to also include the contribution from the CV potential landscape
    double energy_difference(const int spin_index, const Vec3 &initial_Spin,
                             const Vec3 &final_Spin) override;

    // Overrides the accept_move function of MetropolisMCSolver to also update
    // the CV object. This is because CVs are often expensive to recalculate
    // so the algorithm can be optimised by only recalculating if a spin has
    // been accepted.
    void accept_move(const int spin_index, const Vec3 &initial_spin,
                     const Vec3 &final_spin) override;
};

#endif //INCLUDED_JAMS_SOLVERS_CPU_METADYNAMICS_METROPOLIS_SOLVER
