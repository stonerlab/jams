// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <random>
#include <mutex>

#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_array_kernels.h"

#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_array_kernels.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_common.h"
#include "jams/monitors/magnetisation.h"
#include "jams/thermostats/cuda_langevin_arbitrary.h"
#include "jams/thermostats/cuda_langevin_arbitrary_kernel.h"

using namespace std;

namespace {

// convert linear array index (in fftw ordering) into k index (+/-)
// size is the total size of the array
inline int fftw_k_index(const int i, const int size) {
  assert(i < (2 * size - 1));
  if (i < size) {
    return i;
  } else {
    return i - (2 * size - 1);
  }
}

double coth(const double x) {
  return 1 / tanh(x);
}

//// arbitrary correlation function
//double correlator(const double omega) {
//  if (omega == 0.0) return 1.0;
//  return abs(omega)*coth(abs(omega));
//}

// arbitrary correlation function
double correlator(const double omega, const double temperature) {
  if (omega == 0.0) return 1.0;
  double x = (kHBar * abs(omega)) / (kBoltzmann * temperature);
  return 0.5 * x / (exp(x) - 1);
}

double timestep_mismatch_inv_correlator(const double omega, const double bath_time_step) {
  // TODO: check if this should be 1 or 0
  if (omega == 0.0) return 1.0;
  return (0.5 * omega * bath_time_step) / sin(0.5 * omega * bath_time_step);
}

// filter function
double filter(const double omega, const double temperature, const double bath_time_step) {
  auto x = correlator(omega, temperature); // * timestep_mismatch_inv_correlator(omega, bath_time_step);
  assert(!(x < 0.0));
  return sqrt(x);
}

template<typename T1, typename... Args>
vector<T1> discretize_function(std::function<T1(double, Args...)> f, const double delta, const int num_freq, Args... args) {
  vector<T1> result(2 * num_freq - 1);

  for(auto i = 0; i < result.size(); ++i) {
    const auto k = fftw_k_index(i, num_freq);
    result[i] = f(k * delta, args...);
  }

  return result;
}

vector<double> real_discrete_ft(const vector<double> &x) {
  const int num_freq = (x.size() + 1) / 2;
  vector<double> result(x.size());

  for (auto i = 0; i < result.size(); ++i) {
    const auto n = fftw_k_index(i, num_freq);
    double sum = 0.0;
    for (auto j = 0; j < x.size(); ++j) {
      const auto k = fftw_k_index(j, num_freq);
      sum += x[j] * cos(k * n * M_PI / double(num_freq));
    }

    result[i] = sum / double(2 * num_freq);
  }
  return result;
}

}

CudaLangevinArbitraryThermostat::CudaLangevinArbitraryThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  debug_(false),
  filter_temperature_(0.0)
  {
   cout << "\n  initialising CUDA Langevin arbitrary noise thermostat\n";

    num_freq_ = 10000;
    delta_t_ = solver->time_step();
    max_omega_ = kPi / delta_t_;
//    config->lookupValue("thermostat.w_max", max_omega_);

    delta_omega_ = max_omega_ / double(num_freq_);

//   num_freq_ = 100;
//   max_omega_ = 100.0 * kTHz * kTwoPi;
//   config->lookupValue("thermostat.w_max", max_omega_);
//
//   delta_omega_ = max_omega_ / double(num_freq_);
//   delta_t_ = kPi / max_omega_;

   cout << "    max_omega (THz) " << std::fixed << max_omega_ / (kTwoPi * kTHz) << "\n";
   cout << "    delta_t " << std::scientific << delta_t_ << "\n";
   cout << "    num_freq " << num_freq_ << "\n";

   cout << "    initialising CUDA streams\n";

   if (cudaStreamCreate(&dev_stream_) != cudaSuccess) {
     jams_die("Failed to create CUDA stream in CudaLangevinArbitraryThermostat");
   }

   if (cudaStreamCreate(&dev_curand_stream_) != cudaSuccess) {
     jams_die("Failed to create CURAND stream in CudaLangevinArbitraryThermostat");
   }

   cout << "    initialising CURAND\n";

   CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_curand_stream_));

   for (int i = 0; i < num_spins; ++i) {
     for (int j = 0; j < 3; ++j) {
        sigma_(i,j) = sqrt((2.0 * globals::alpha(i) * kBoltzmann) / (globals::mus(i) * kBohrMagneton * kGyromagneticRatio));
     }
   }

   white_noise_.resize((num_spins * 3) * (2*num_freq_ - 1));

   CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), white_noise_.device_data(), white_noise_.size(), 0.0, 1.0));


    debug_file_.open("noise.tsv");
  }

void CudaLangevinArbitraryThermostat::update() {

  if (filter_temperature_ != this->temperature()) {
    filter_temperature_ = this->temperature();
    auto discrete_filter = discretize_function(std::function<double(double, double, double)>(filter), delta_omega_, num_freq_, temperature_, delta_t_);
    auto convoluted_filter = real_discrete_ft(discrete_filter);
    filter_.resize(convoluted_filter.size());
    std::copy(convoluted_filter.begin(), convoluted_filter.end(), filter_.begin());
  }
  assert(filter_.size() != 0);

  int block_size = 256;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  const double temperature = this->temperature();

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(),
                                      dev_curand_stream_));

  const auto n = pbc(solver->iteration(), (2 * num_freq_ - 1));
  arbitrary_stochastic_process_cuda_kernel<<<grid_size, block_size, 0, dev_stream_ >>>(
      noise_.device_data(),
      filter_.device_data(),
      white_noise_.device_data(),
      delta_t_,
      n,
      num_freq_,
      globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;


  // scale by sigma
  // TODO: does temperature need to go here or in the kernel above?
  cuda_array_elementwise_scale(globals::num_spins, 3, sigma_.device_data(), temperature, noise_.device_data(), 1, noise_.device_data(), 1, dev_stream_);


  debug_file_ << solver->iteration() * delta_t_ << " " << noise_(0, 0)
              << "\n";

  // generate new random numbers
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(jams::instance().curand_generator(),
                                 white_noise_.device_data() +
                                 globals::num_spins3 *
                                 pbc(solver->iteration() + num_freq_,
                                     2 * num_freq_ - 1),
                                 globals::num_spins3, 0.0, 1.0));
}

CudaLangevinArbitraryThermostat::~CudaLangevinArbitraryThermostat() {
  debug_file_.close();

  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }

  if (dev_curand_stream_ != nullptr) {
    cudaStreamDestroy(dev_curand_stream_);
  }
}
