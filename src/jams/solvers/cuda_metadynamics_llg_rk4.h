// cuda_metadynamics_llg_rk4.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_SOLVE_CUDA_METADYNAMICS_LLG_RK4
#define INCLUDED_JAMS_SOLVE_CUDA_METADYNAMICS_LLG_RK4


#if HAS_CUDA

#include <jams/solvers/cuda_llg_rk4.h>
#include "jams/metadynamics/metadynamics_potential.h"
#include <fstream>
#include <memory>
#include <vector>

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDAMetadynamicsLLGRK4Solver : public CUDALLGRK4Solver {
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

    /// Output file for metadynamics statistics
    std::ofstream metadynamics_stats_file_;

    jams::MultiArray<double,2> metadynamics_fields_;

public:
    /// Default constructor
    CUDAMetadynamicsLLGRK4Solver() = default;

    /// Default destructor
    ~CUDAMetadynamicsLLGRK4Solver() override = default;

    // Initializes the CUDAMetadynamicsLLGRK4Solver using settings from the global
    // config. This will also create and attach a coordinate variable object
    // specified in the config.
    void initialize(const libconfig::Setting& settings) override;

    // Runs the metadynamics solver. This involves running the LLG solver and
    // also triggering the insertion of gaussians into the potential
    // energy landscape.
    void run() override;

    void compute_fields() override;


    std::string name() const override { return "llg-metadynamics-rk4-gpu"; }

};

#endif

#endif //INCLUDED_JAMS_SOLVE_CUDA_METADYNAMICS_LLG_RK4




