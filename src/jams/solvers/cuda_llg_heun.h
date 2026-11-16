// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_H
#define JAMS_SOLVER_CUDA_HEUNLLG_H

#if HAS_CUDA

#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDAHeunLLGSolver : public CudaSolver {
 public:
  CUDAHeunLLGSolver() = default;
  ~CUDAHeunLLGSolver() override = default;

  inline explicit CUDAHeunLLGSolver(const libconfig::Setting& settings) { initialize(settings); }

  void initialize(const libconfig::Setting& settings) override;
  void run() override;

  std::string name() const override { return "llg-heun-gpu"; }

  bool supports_deterministic_spin_rate() const override { return true; }
  double max_deterministic_spin_rate() override { return max_llg_spin_rate_cuda(gyro_eff_); }

 private:
  jams::MultiArray<jams::Real, 1> gyro_eff_;
  jams::MultiArray<double, 2> spins_initial_;
};

#endif

#endif  // JAMS_SOLVER_CUDA_HEUNLLG_H
