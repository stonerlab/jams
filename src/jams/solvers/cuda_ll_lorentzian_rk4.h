// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_LL_LORENTZIAN_RK4_H
#define JAMS_SOLVER_CUDA_LL_LORENTZIAN_RK4_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDALLLorentzianRK4Solver : public CudaSolver {
  public:
    CUDALLLorentzianRK4Solver() = default;
    ~CUDALLLorentzianRK4Solver() override = default;

    inline explicit CUDALLLorentzianRK4Solver(const libconfig::Setting &settings) {
      initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "ll-lorentzian-rk4-gpu"; }

  private:
    CudaStream dev_stream_;

    double lorentzian_omega_;
    double lorentzian_gamma_;
    double lorentzian_A_;

    jams::MultiArray<double, 2> s_old_;
    jams::MultiArray<double, 2> s_k1_;
    jams::MultiArray<double, 2> s_k2_;
    jams::MultiArray<double, 2> s_k3_;
    jams::MultiArray<double, 2> s_k4_;


    jams::MultiArray<double, 2> w_memory_process_;
    jams::MultiArray<double, 2> w_memory_process_old_;
    jams::MultiArray<double, 2> w_memory_process_k1_;
    jams::MultiArray<double, 2> w_memory_process_k2_;
    jams::MultiArray<double, 2> w_memory_process_k3_;
    jams::MultiArray<double, 2> w_memory_process_k4_;

    jams::MultiArray<double, 2> v_memory_process_;
    jams::MultiArray<double, 2> v_memory_process_old_;
    jams::MultiArray<double, 2> v_memory_process_k1_;
    jams::MultiArray<double, 2> v_memory_process_k2_;
    jams::MultiArray<double, 2> v_memory_process_k3_;
    jams::MultiArray<double, 2> v_memory_process_k4_;

};

#endif

#endif // JAMS_SOLVER_CUDA_LL_LORENTZIAN_RK4_H

