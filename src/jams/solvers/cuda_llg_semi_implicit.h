#ifndef JAMS_SOLVER_CUDA_LLG_SemiImplict_H
#define JAMS_SOLVER_CUDA_LLG_SemiImplict_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDALLGSemiImplictSolver : public CudaSolver {
public:
    inline explicit CUDALLGSemiImplictSolver(const libconfig::Setting &settings) {
        initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "llg-simp-gpu"; }
private:
    CudaStream dev_stream_;
    jams::MultiArray<double, 2> s_init_;
    jams::MultiArray<double, 2> s_pred_;

};

#endif

#endif // JAMS_SOLVER_CUDA_LLG_SemiImplict_H

