//
// Created by Joseph Barker on 05/01/2026.
//

#ifndef JAMS_CUDA_LLG_RKMK2_H
#define JAMS_CUDA_LLG_RKMK2_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDALLGRKMK2Solver : public CudaSolver {
public:
    inline explicit CUDALLGRKMK2Solver(const libconfig::Setting &settings) {
        initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "llg-rkmk2-gpu"; }
private:
    CudaStream dev_stream_;
    jams::MultiArray<double, 2> s_init_;
    jams::MultiArray<double, 2> phi_;

};

#endif


#endif //JAMS_CUDA_LLG_RKMK2_H