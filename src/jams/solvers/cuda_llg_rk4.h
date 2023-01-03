// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_LLG_RK4_H
#define JAMS_SOLVER_CUDA_LLG_RK4_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDALLGRK4Solver : public CudaSolver {
  public:
    CUDALLGRK4Solver() = default;
    ~CUDALLGRK4Solver() override = default;

    inline explicit CUDALLGRK4Solver(const libconfig::Setting &settings) {
      initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "llg-rk4-gpu"; }

  private:
    CudaStream dev_stream_;

    jams::MultiArray<double, 2> s_old_;
    jams::MultiArray<double, 2> k1_;
    jams::MultiArray<double, 2> k2_;
    jams::MultiArray<double, 2> k3_;
    jams::MultiArray<double, 2> k4_;
};

#endif

#endif // JAMS_SOLVER_CUDA_LLG_RK4_H

