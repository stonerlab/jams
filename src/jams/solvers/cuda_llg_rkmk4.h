//
// Created by Joseph Barker on 05/01/2026.
//

#ifndef JAMS_CUDA_LLG_RKMK4_H
#define JAMS_CUDA_LLG_RKMK4_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDALLGRKMK4Solver : public CudaSolver {
public:
    inline explicit CUDALLGRKMK4Solver(const libconfig::Setting &settings) {
        initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "llg-rkmk4-gpu"; }
private:
    CudaStream dev_stream_;
    jams::MultiArray<double, 2> s_init_;
    jams::MultiArray<double, 2> k1_;
    jams::MultiArray<double, 2> k2_;
    jams::MultiArray<double, 2> k3_;
};

#endif

#endif //JAMS_CUDA_LLG_RKMK4_H
