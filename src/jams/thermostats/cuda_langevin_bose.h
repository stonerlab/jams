// Copyright 2014 Joseph Barker. All rights reserved.

// This thermostat implementation is designed to reproduce a semiquantum thermostat
// which has a coth(omega) frequency dependence.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H

#if HAS_CUDA

#include <curand.h>
#include <fstream>
#include <mutex>

#include "jams/core/thermostat.h"

#include "jblib/containers/cuda_array.h"

class CudaLangevinBoseThermostat : public Thermostat {
 public:
  CudaLangevinBoseThermostat(const double &temperature, const double &sigma, const int num_spins);
  ~CudaLangevinBoseThermostat();

  void update();

  // override the base class implementation
  const double* noise() { return dev_noise_.data(); }

 private:

    void warmup(const unsigned steps);

    bool debug_;
    bool is_warmed_up_ = false;
    unsigned num_warm_up_steps_ = 0;

    jblib::CudaArray<double, 1> dev_noise_;
    jblib::CudaArray<double, 1> dev_zeta0_;
    jblib::CudaArray<double, 1> dev_zeta5_;
    jblib::CudaArray<double, 1> dev_zeta5p_;
    jblib::CudaArray<double, 1> dev_zeta6_;
    jblib::CudaArray<double, 1> dev_zeta6p_;

    jblib::CudaArray<double, 1> dev_eta0_;
    jblib::CudaArray<double, 1> dev_eta1a_;
    jblib::CudaArray<double, 1> dev_eta1b_;
    jblib::CudaArray<double, 1> dev_sigma_;
    curandGenerator_t           dev_rng_ = nullptr;  // device random generator
    cudaStream_t                dev_stream_ = nullptr;
    cudaStream_t                dev_curand_stream_ = nullptr;
    double                      delta_tau_;
    double                      omega_max_;
    std::ofstream               outfile_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_H
