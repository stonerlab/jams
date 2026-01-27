// cuda_llg_dm.h

#ifndef JAMS_CUDA_LLG_DM_H
#define JAMS_CUDA_LLG_DM_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDALLGDMSolver : public CudaSolver {
public:
    inline explicit CUDALLGDMSolver(const libconfig::Setting &settings) { initialize(settings); }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "llg-dm-gpu"; }

private:
    CudaStream dev_stream_;
    jams::MultiArray<double, 2> s_init_;
    jams::MultiArray<double, 2> s_pred_;
    jams::MultiArray<double, 2> omega1_;   // store ω_n for averaging
};

#endif
#endif // JAMS_CUDA_LLG_DM_H