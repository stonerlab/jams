// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_LLG_RK4_H
#define JAMS_SOLVER_CUDA_LLG_RK4_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/containers/multiarray.h"
#include "jams/solvers/cuda_rk4_base.h"

class CUDALLGRK4Solver : public CudaRK4BaseSolver {
  public:
    explicit CUDALLGRK4Solver(const libconfig::Setting &settings);

    std::string name() const override { return "llg-rk4-gpu"; }

    void function_kernel(jams::MultiArray<double, 2>& spins, jams::MultiArray<double, 2>& k) override;
    void post_step(jams::MultiArray<double, 2>& spins) override;

  private:
    jams::MultiArray<double, 2> extra_torque_;
};

#endif

#endif // JAMS_SOLVER_CUDA_LLG_RK4_H
