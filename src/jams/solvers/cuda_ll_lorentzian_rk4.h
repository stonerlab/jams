// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_LL_LORENTZIAN_RK4_H
#define JAMS_SOLVER_CUDA_LL_LORENTZIAN_RK4_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"

class CUDALLLorentzianRK4Solver : public CudaSolver {
  public:
    CUDALLLorentzianRK4Solver() = default;
    ~CUDALLLorentzianRK4Solver() = default;
    void initialize(const libconfig::Setting& settings);
    void run();

  private:
    CudaStream dev_stream_;
    double dt_;

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

