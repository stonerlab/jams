#ifndef JAMS_SOLVER_CUDA_RK4_LLG_SOT_H
#define JAMS_SOLVER_CUDA_RK4_LLG_SOT_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/containers/multiarray.h"
#include "jams/solvers/cuda_rk4_base.h"

class CudaRK4LLGSOTSolver : public CudaRK4BaseSolver {
public:
    explicit CudaRK4LLGSOTSolver(const libconfig::Setting &settings);

    std::string name() const override { return "llg-sot-rk4-gpu"; }

    void function_kernel(jams::MultiArray<double, 2>& spins, jams::MultiArray<double, 2>& k) override;
    void post_step(jams::MultiArray<double, 2>& spins) override;

private:
    jams::MultiArray<double, 2> spin_polarisation_;
    jams::MultiArray<double, 1> sot_coefficient_;
};

#endif

#endif // JAMS_SOLVER_CUDA_RK4_LLG_SOT_H

