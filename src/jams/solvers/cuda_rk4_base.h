#ifndef INCLUDED_JAMS_CUDA_RK4_BASE
#define INCLUDED_JAMS_CUDA_RK4_BASE

#if HAS_CUDA

#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"
#include "jams/core/thermostat.h"

class CudaRK4BaseSolver : public CudaSolver {
 public:
  CudaRK4BaseSolver() = default;
  ~CudaRK4BaseSolver() override = default;

  explicit CudaRK4BaseSolver(const libconfig::Setting& settings,
                             jams::ThermalTorque torque = jams::ThermalTorque::llg)
      : thermal_torque_(torque) { initialize(settings); }

  void initialize(const libconfig::Setting& settings) override;
  void run() override;

 protected:
  // Prepare the stage spins before evaluating the Hamiltonians.
  virtual void pre_stage(jams::MultiArray<double, 2>& spins, cudaStream_t stream) {}

  // Evaluate the RHS with the stage fields and noise ready on stream.
  virtual void function_kernel(jams::MultiArray<double, 2>& spins, jams::MultiArray<double, 2>& k,
                               const jams::Real* noise, cudaStream_t stream) = 0;

  // Finish updating the accepted spins on stream.
  virtual void post_step(jams::MultiArray<double, 2>& spins, cudaStream_t stream) {}

  jams::MultiArray<jams::Real, 1> gyro_eff_;

 private:
  jams::ThermalTorque thermal_torque_ = jams::ThermalTorque::llg;

  void evaluate_stage(jams::MultiArray<double, 2>& k, const jams::Real* noise, cudaStream_t stream);

  jams::MultiArray<double, 2> spins_initial_;
  jams::MultiArray<double, 2> k1_;
  jams::MultiArray<double, 2> k2_;
  jams::MultiArray<double, 2> k3_;
  jams::MultiArray<double, 2> k4_;
};

#endif

#endif
