// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <random>
#include <mutex>

#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_array_kernels.h"

#include "cuda_langevin_bose.h"
#include "cuda_langevin_bose_kernel.h"

#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/random.h"
#include "jams/helpers/error.h"

#include "jams/monitors/magnetisation.h"

using namespace std;

CudaLangevinBoseThermostat::CudaLangevinBoseThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  debug_(false),
  dev_noise_(3 * num_spins, 0.0),
  dev_zeta0_(4 * num_spins * 3, 0.0),
  dev_zeta5_(num_spins * 3, 0.0),
  dev_zeta5p_(num_spins * 3, 0.0),
  dev_zeta6_(num_spins * 3, 0.0),
  dev_zeta6p_(num_spins * 3, 0.0),
  dev_eta0_(4 * num_spins * 3, 0.0),
  dev_eta1a_(2 * num_spins * 3, 0.0),
  dev_eta1b_(2 * num_spins * 3, 0.0),
  dev_sigma_(num_spins, 0.0)
 {
   cout << "\n  initialising CUDA Langevin semi-quantum noise thermostat\n";

   debug_ = false;

   if(debug_) {
     debug_noise_outfile_.open(seedname + "_qnoise.tsv");
     debug_noise_outfile_ << "time\tnoise0\tnoise1\tnoise2\tnoise3\tnoise4\tnoise5\tnoise6\tnoise7\tnoise8\tnoise9\n";
   }

   config->lookupValue("thermostat.zero_point", do_zero_point_);

   double t_warmup = 1e-10; // 0.1 ns
   config->lookupValue("thermostat.warmup_time", t_warmup);

   omega_max_ = 25.0 * kTwoPi * kTHz;
   config->lookupValue("thermostat.w_max", omega_max_);

   double dt_thermostat = ::config->lookup("solver.t_step");
   delta_tau_ = (dt_thermostat * kBoltzmann) / kHBar;

   uint64_t dev_rng_seed = jams::random_generator()();

   cout << "    seed " << dev_rng_seed << "\n";
   cout << "    omega_max (THz) " << omega_max_ / (kTwoPi * kTHz) << "\n";
   cout << "    hbar*w/kB " << (kHBar * omega_max_) / (kBoltzmann) << "\n";
   cout << "    t_step " << dt_thermostat << "\n";
   cout << "    delta tau " << delta_tau_ << "\n";

   cout << "    initialising CUDA streams\n";

   if (cudaStreamCreate(&dev_stream_) != cudaSuccess) {
     die("Failed to create CUDA stream in CudaLangevinBoseThermostat");
   }

   if (cudaStreamCreate(&dev_curand_stream_) != cudaSuccess) {
     die("Failed to create CURAND stream in CudaLangevinBoseThermostat");
   }

   cout << "    initialising CURAND\n";

   // initialize and seed the CURAND generator on the device
   if (curandCreateGenerator(&dev_rng_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
     die("Failed to create CURAND generator in CudaLangevinBoseThermostat");
   }

   // initialize zeta and eta with random variables
   curandSetStream(dev_rng_, dev_curand_stream_);

   cout << "    seeding CURAND " << dev_rng_seed << "\n";

   if (curandSetPseudoRandomGeneratorSeed(dev_rng_, dev_rng_seed) != CURAND_STATUS_SUCCESS) {
     die("Failed to set CURAND seed in CudaLangevinBoseThermostat");
   }

   if (curandGenerateSeeds(dev_rng_) != CURAND_STATUS_SUCCESS) {
     die("Failed to generate CURAND seeds in CudaLangevinBoseThermostat");
   }

   cout << "    allocating GPU memory\n";

   if (curandGenerateNormalDouble(dev_rng_, dev_eta0_.data(), dev_eta0_.size(), 0.0, 1.0)
       != CURAND_STATUS_SUCCESS) {
     die("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::constructor");
   }

   if (curandGenerateNormalDouble(dev_rng_, dev_eta1a_.data(), dev_eta1a_.size(), 0.0, 1.0)
       != CURAND_STATUS_SUCCESS) {
     die("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::constructor");
   }

   if (curandGenerateNormalDouble(dev_rng_, dev_eta1b_.data(), dev_eta1b_.size(), 0.0, 1.0)
       != CURAND_STATUS_SUCCESS) {
     die("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::constructor");
   }

   jblib::Array<double, 2> scale(num_spins, 3);
   for (int i = 0; i < num_spins; ++i) {
     for (int j = 0; j < 3; ++j) {
       scale(i, j) = (kBoltzmann) *
                     sqrt((2.0 * globals::alpha(i) * globals::mus(i)) / (kHBar * kGyromagneticRatio * kBohrMagneton));
     }
   }

   dev_sigma_ = jblib::CudaArray<double, 1>(scale);

   num_warm_up_steps_ = static_cast<unsigned>(t_warmup / dt_thermostat);
 }

void CudaLangevinBoseThermostat::update() {
  if (!is_warmed_up_) {
    is_warmed_up_ = true;
    warmup(num_warm_up_steps_);
  }

  int block_size = 96;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  swap(dev_eta1a_, dev_eta1b_);
  curandGenerateNormalDouble(dev_rng_, dev_eta1a_.data(), dev_eta1a_.size(), 0.0, 1.0);

  bose_coth_stochastic_process_cuda_kernel<<<grid_size, block_size, 0, dev_stream_ >>> (
    dev_noise_.data(),
    dev_zeta5_.data(),
    dev_zeta5p_.data(),
    dev_zeta6_.data(),
    dev_zeta6p_.data(),
    dev_eta1b_.data(),
    dev_sigma_.data(),
    delta_tau_ * this->temperature(),
    this->temperature(),
    (kHBar * omega_max_) / (kBoltzmann * this->temperature()),  // w_m
    globals::num_spins3);

  if (do_zero_point_) {
    curandGenerateNormalDouble(dev_rng_, dev_eta0_.data(), dev_eta0_.size(), 0.0, 1.0);

    bose_zero_point_stochastic_process_cuda_kernel << < grid_size, block_size, 0, dev_stream_ >> > (
            dev_noise_.data(),
                    dev_zeta0_.data(),
                    dev_eta0_.data(),
                    dev_sigma_.data(),
                    delta_tau_ * this->temperature(),
                    this->temperature(),
                    (kHBar * omega_max_) / (kBoltzmann * this->temperature()),  // w_m
                    globals::num_spins3);
  }

  if (debug_ && is_warmed_up_) {
    dev_noise_.copy_to_host_array(noise_);
    debug_noise_outfile_ << solver->time() << "\t";
    for (auto i = 0; i < 10; ++i) {
      debug_noise_outfile_ << noise_[i] << "\t";
    }
    debug_noise_outfile_ << std::endl;
  }
}

CudaLangevinBoseThermostat::~CudaLangevinBoseThermostat() {
  if (dev_rng_ != nullptr) {
    curandDestroyGenerator(dev_rng_);
  }

  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }

  if (dev_curand_stream_ != nullptr) {
    cudaStreamDestroy(dev_curand_stream_);
  }
}

void CudaLangevinBoseThermostat::warmup(const unsigned steps) {
  cout << "warming up thermostat " << steps << " steps @ " << this->temperature() << "K" << std::endl;

  for (auto i = 0; i < steps; ++i) {
    update();
  }
}
