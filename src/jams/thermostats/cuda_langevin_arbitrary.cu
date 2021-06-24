// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <random>
#include <mutex>
#include <fstream>

#include "jams/helpers/output.h"
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

//#define PRINT_NOISE

using namespace std;

namespace {

double coth(const double x) {
  return 1 / tanh(x);
}

//// arbitrary correlation function
//double correlator(const double omega) {
//  if (omega == 0.0) return 1.0;
//  return abs(omega)*coth(abs(omega));
//}

double barker_correlator(const double omega, const double temperature) {
  if (omega == 0.0) return 1.0;
  double x = (kHBar * abs(omega)) / (kBoltzmann * temperature);
  return x / (exp(x) - 1);
}

double zero_point_correlator(const double omega, const double temperature) {
  if (omega == 0.0) return 1.0;
  double x = (kHBar * abs(omega)) / (2.0 * kBoltzmann * temperature);
  return x * coth(x);
}

double lorentzian_correlator(const double omega, const double temperature) {
  if (omega == 0.0) return 1.0;

  double Gamma = 3.71e12 * kTwoPi;
  double omega_0 = 6.27e12 * kTwoPi;

  double lorentzian = pow4(omega_0) / (pow2(pow2(omega_0) - pow2(omega)) + pow2(omega * Gamma));


  double x = (kHBar * abs(omega)) / (2.0 * kBoltzmann * temperature);
  return lorentzian * (x * coth(x));
}

double timestep_mismatch_inv_correlator(const double omega, const double bath_time_step) {
  // TODO: check if this should be 1 or 0
  if (omega == 0.0) return 1.0;
  return (0.5 * omega * bath_time_step) / sin(0.5 * omega * bath_time_step);
}

// filter function
double barker_filter(const double omega, const double temperature, const double bath_time_step) {
  auto x = barker_correlator(omega, temperature); // * timestep_mismatch_inv_correlator(omega, bath_time_step);
  assert(!(x < 0.0));
  return sqrt(x);
}

// filter function
double zero_point_filter(const double omega, const double temperature, const double bath_time_step) {
  auto x = zero_point_correlator(omega, temperature); // * timestep_mismatch_inv_correlator(omega, bath_time_step);
  assert(!(x < 0.0));
  return sqrt(x);
}

// filter function
double lorentzian_filter(const double omega, const double temperature, const double bath_time_step) {
  auto x = lorentzian_correlator(omega, temperature); // * timestep_mismatch_inv_correlator(omega, bath_time_step);
  assert(!(x < 0.0));
  return sqrt(x);
}

template<typename T1, typename... Args>
vector<T1> discretize_function(std::function<T1(double, Args...)> f, const double delta, const int num_freq, Args... args) {
  vector<T1> result(num_freq);

  for(auto i = 0; i < result.size(); ++i) {
    result[i] = f(i * delta, args...);
  }

  return result;
}

// Because we are doing a real fourier transform only the positive frequencies
// are calculated and stored.
vector<double> real_discrete_ft(vector<double> &x) {

  int size = static_cast<int>(x.size());
  fftw_plan plan = fftw_plan_r2r_1d(
      size, //int n
      x.data(), // double *in
      x.data(), // double *out
      FFTW_REDFT01, // fftw_r2r_kind kind
      FFTW_ESTIMATE); // unsigned flags

  fftw_execute(plan);
  fftw_destroy_plan(plan);

  // normalise by the logical size of the DFT
  // (see http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029)
  for (double &i : x) {
    i /= 2*size;
  }

  return x;
}

}

CudaLangevinArbitraryThermostat::CudaLangevinArbitraryThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  debug_(false),
  filter_temperature_(0.0) {

  cout << "\n  initialising CUDA Langevin arbitrary noise thermostat\n";

  num_freq_ = jams::config_optional(config->lookup("thermostat"), "num_freq", 4000);
  num_trunc_ = jams::config_optional(config->lookup("thermostat"), "num_trunc", 2000);

  assert(num_trunc_ <= num_freq_);

  delta_t_ = solver->time_step();
  max_omega_ = kPi / delta_t_;
  delta_omega_ = max_omega_ / double(num_freq_);

  config->lookupValue("thermostat.zero_point", do_zero_point_);

  cout << "    max_omega (THz) " << std::fixed << max_omega_ / (kTwoPi * kTHz) << "\n";
  cout << "    delta_omega (THz) " << std::fixed << delta_omega_ / (kTwoPi * kTHz) << "\n";
  cout << "    delta_t " << std::scientific << delta_t_ << "\n";
  cout << "    num_freq " << num_freq_ << "\n";
  cout << "    num_trunc " << num_trunc_ << "\n";

  cout << "    initialising CUDA streams\n";

  if (cudaStreamCreate(&dev_stream_) != cudaSuccess) {
   jams_die("Failed to create CUDA stream in CudaLangevinArbitraryThermostat");
  }

  if (cudaStreamCreate(&dev_curand_stream_) != cudaSuccess) {
   jams_die("Failed to create CURAND stream in CudaLangevinArbitraryThermostat");
  }

  cout << "    initialising CURAND\n";

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_curand_stream_));


  // If the noise term is written as a B-field in Tesla it should look like:
  //
  // B(t) = \sqrt{T} \sigma_i \xi(t, T)
  //
  // where the \sigma_i depends on parameters on the local site and \xi
  // is a dimensionless noise.
  //
  // \sigma_i = \sqrt((2 \alpha_i k_B) / (\gamma_i \mu_{s,i}))
  //
  // where:
  // - \alpha_i is a coupling parameter to the bath
  // - k_B is Boltzmann constant
  // - \gamma_i is the local gyromagnetic ratio (usually g_e \mu_B / \hbar)
  // - \mu_{s,i} is the local magnetic moment in Bohr magnetons
  //
  // Inside of JAMS we have to take into account the reduce units namely that
  // globals::gyro(i) == -\gamma_i / (\mu_{s,i}*kGyromagneticRatio), hence we
  // should convert back to get \gamma_i.
  //
  // We also need to account for the fact that all fields are multiplied by
  // globals::gyro(i) in the LLG in JAMS. In principle we want a B-field in
  // tesla to be multiplied by \gamma_i, but the \mu_{s,i} from the field
  // derivative H = -(1/\mu_{s,i}) \partial \mathcal{H}/\partial \vec{S}_i
  // is factored out into the LLG. Hence our thermostat must be premultiplied
  // by globals::mus(i) to account for this.
  //
  // Finally we put the 1/\sqrt{\Delta t} which is the scaling of the Gaussian
  // noise processes into sigma (because it's a convenient place for it).
  //
  for (int i = 0; i < num_spins; ++i) {
   // globals::gyro(i) = - gamma / mus
   const auto gamma = std::abs(globals::gyro(i)) * globals::mus(i) * kGyromagneticRatio;
   // thermostat fields are treated like other fields in the LLG kernels and are multiplied by globals::gyro (i.e. gamma/mus) hence we want to pre multiply
   // sigma by (1/mus)
   for (int j = 0; j < 3; ++j) {
      sigma_(i,j) = sqrt((2.0 * globals::alpha(i) * globals::mus(i) * kBoltzmann) / (kBohrMagneton * gamma  * solver->time_step()));
   }
  }

  white_noise_.resize((num_spins * 3) * (2*num_trunc_ + 1));

  CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), white_noise_.device_data(), white_noise_.size(), 0.0, 1.0));

  #ifdef PRINT_NOISE
  debug_file_.open("noise.tsv");
  #endif

}

void CudaLangevinArbitraryThermostat::update() {

  if (filter_temperature_ != this->temperature() || solver->iteration() == 0) {
    filter_temperature_ = this->temperature();

    vector<double> discrete_filter;
    if (do_zero_point_) {
      discrete_filter = discretize_function(std::function<double(double, double, double)>(
          lorentzian_filter), delta_omega_, num_freq_, temperature_, delta_t_);
    } else {
      discrete_filter = discretize_function(std::function<double(double, double, double)>(
          barker_filter), delta_omega_, num_freq_, temperature_, delta_t_);
    }

    auto convoluted_filter = real_discrete_ft(discrete_filter);
//    filter_.resize(convoluted_filter.size());
//    std::copy(convoluted_filter.begin(), convoluted_filter.end(), filter_.begin());

    std::ofstream filter_full_file(jams::output::full_path_filename("noise_filter_full.tsv"));
    for (auto i = 0; i < convoluted_filter.size(); ++i) {
      filter_full_file << i << " " << convoluted_filter[i] << "\n";
    }
    filter_full_file.close();

    filter_.resize(num_trunc_);
    std::copy(convoluted_filter.begin(), convoluted_filter.begin()+num_trunc_, filter_.begin());

    std::ofstream filter_file(jams::output::full_path_filename("noise_filter_trunc.tsv"));
    for (auto i = 0; i < filter_.size(); ++i) {
      filter_file << i << " " << filter_(i) << "\n";
    }
    filter_file.close();
  }

  assert(filter_.size() != 0);

  int block_size = 128;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  const double temperature = this->temperature();

  CHECK_CURAND_STATUS(
      curandSetStream(jams::instance().curand_generator(),dev_curand_stream_));

  const auto n = pbc(solver->iteration(), (2 * num_trunc_ + 1));
  arbitrary_stochastic_process_cuda_kernel<<<grid_size, block_size, 0, dev_stream_ >>>(
      noise_.device_data(),
      filter_.device_data(),
      white_noise_.device_data(),
      n,
      num_trunc_,
      globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;


  // scale by sigma
  cuda_array_elementwise_scale(globals::num_spins, 3, sigma_.device_data(), sqrt(temperature), noise_.device_data(), 1, noise_.device_data(), 1, dev_stream_);


  #ifdef PRINT_NOISE
    debug_file_ << solver->iteration() * delta_t_ << " " << noise_(0, 0)
                << "\n";
  #endif

  // generate new random numbers
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(jams::instance().curand_generator(),
                                 white_noise_.device_data() +
                                 globals::num_spins3 *
                                 pbc(solver->iteration() + num_trunc_,
                                     2 * num_trunc_ + 1),
                                 globals::num_spins3, 0.0, 1.0));
}

CudaLangevinArbitraryThermostat::~CudaLangevinArbitraryThermostat() {
  #ifdef PRINT_NOISE
    debug_file_.close();
  #endif

  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }

  if (dev_curand_stream_ != nullptr) {
    cudaStreamDestroy(dev_curand_stream_);
  }
}

#undef PRINT_NOISE