// thm_bose_einstein_cuda_srk4.h                                       -*-C++-*-

#ifndef INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4
#define INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4

#if HAS_CUDA

#include "jams/core/noise_generator.h"
#include "jams/cuda/cuda_stream.h"
#include "jams/containers/multiarray.h"

struct BoseEinsteinCudaSRK4NoiseGeneratorConfig {
  double warmup_time_ps = 100.0;
};

namespace jams {

class BoseEinsteinCudaSRK4NoiseGenerator : public NoiseGenerator {
 public:
  BoseEinsteinCudaSRK4NoiseGenerator(const double& temperature,
                                     const double timestep,
                                     int num_vectors,
                                     const BoseEinsteinCudaSRK4NoiseGeneratorConfig& config = {});
  ~BoseEinsteinCudaSRK4NoiseGenerator() override;

  void update() override;

 private:
  void warmup(unsigned steps);

  bool is_warmed_up_ = false;
  unsigned num_warm_up_steps_ = 0;
  double delta_tau_;
  int num_channels_ = 0;

  jams::MultiArray<double, 1> w5_;
  jams::MultiArray<double, 1> v5_;
  jams::MultiArray<double, 1> w6_;
  jams::MultiArray<double, 1> v6_;
  jams::MultiArray<double, 1> psi5_;
  jams::MultiArray<double, 1> psi6_;

  cudaEvent_t event5_{};
  cudaEvent_t event6_{};

  curandGenerator_t prng5_{};
  curandGenerator_t prng6_{};

  CudaStream dev_stream5_;
  CudaStream dev_stream6_;
};

}  // namespace jams

#endif  // CUDA
#endif  // INCLUDED_JAMS_THM_BOSE_EINSTEIN_CUDA_SRK4
