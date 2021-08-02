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
//#define PRINT_MEMORY

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

double barker_correlator(double& omega, double& temperature) {
  if (omega == 0.0) return 1.0;
  double x = (kHBarIU * abs(omega)) / (kBoltzmannIU * temperature);
  return x / (exp(x) - 1);
}

double zero_point_correlator(double& omega, double& temperature) {
  if (omega == 0.0) return 1.0;
  double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return x * coth(x);
}

double lorentzian_correlator(double& omega, double& temperature, double& gamma, double& omega0, double& A) {
  if (omega == 0.0) return 1.0;

  double lorentzian = (A * gamma) / (pow2(pow2(omega0) - pow2(omega)) + pow2(omega * gamma));

  double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return lorentzian * (x * coth(x));
}

double timestep_mismatch_inv_correlator(const double omega, const double bath_time_step) {
  // TODO: check if this should be 1 or 0
  if (omega == 0.0) return 1.0;
  return (0.5 * omega * bath_time_step) / sin(0.5 * omega * bath_time_step);
}

template<typename... Args>
vector<double> discrete_psd_filter(
    std::function<double(double&, Args...)> const& spectral_function
    , const double& delta_omega
    , const int& num_freq
    , const double& bath_time_step
    , Args&&... args) {

  vector<double> result(num_freq);

  for(auto i = 0; i < result.size(); ++i) {
    double omega = i * delta_omega;
    auto psd = spectral_function(omega, args...); // * timestep_mismatch_inv_correlator(omega, bath_time_step);
    assert(!(x < 0.0));
    result[i] = sqrt(psd);
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

  if (lattice->num_materials() > 1) {
    throw std::runtime_error("CudaLangevinArbitraryThermostat is only implemented for single material cells");
  }
  cout << "\n  initialising CUDA Langevin arbitrary noise thermostat\n";


  lorentzian_gamma_ = jams::config_required<double>(config->lookup("thermostat"), "lorentzian_gamma"); // 3.71e12 * kTwoPi;
  lorentzian_omega0_ = jams::config_required<double>(config->lookup("thermostat"), "lorentzian_omega0"); // 6.27e12 * kTwoPi;

  lorentzian_A_ =  (globals::alpha(0) * pow4(lorentzian_omega0_)) / (lorentzian_gamma_);

  zero(memory_w_process_.resize(globals::num_spins, 3));
  zero(memory_v_process_.resize(globals::num_spins, 3));

  for (auto i = 0; i < num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      memory_w_process_(i,j) = lorentzian_A_ * globals::gyro(i) * globals::s(i,j);
    }
  }


  num_freq_ = jams::config_optional(config->lookup("thermostat"), "num_freq", 10000);
  num_trunc_ = jams::config_optional(config->lookup("thermostat"), "num_trunc", 2000);

  assert(num_trunc_ <= num_freq_);

  delta_t_ = solver->time_step();
  max_omega_ = kPi / delta_t_;
  delta_omega_ = max_omega_ / double(num_freq_);

  config->lookupValue("thermostat.zero_point", do_zero_point_);

  cout << "    lorentzian gamma (THz) " << std::fixed << lorentzian_gamma_ / (kTwoPi) << "\n";
  cout << "    lorentzian omega0 (THz) " << std::fixed << lorentzian_omega0_ / (kTwoPi) << "\n";
  cout << "    lorentzian A " << std::fixed << lorentzian_A_ << "\n";

  cout << "    max_omega (THz) " << std::fixed << max_omega_ / (kTwoPi) << "\n";
  cout << "    delta_omega (THz) " << std::fixed << delta_omega_ / (kTwoPi) << "\n";
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
  // We put the 1/\sqrt{\Delta t} which is the scaling of the Gaussian
  // noise processes into sigma (because it's a convenient place for it).
  //
  for (int i = 0; i < num_spins; ++i) {
   for (int j = 0; j < 3; ++j) {
     sigma_(i, j) = sqrt((2.0 * kBoltzmannIU) /
                         (globals::mus(i) * globals::gyro(i) * solver->time_step()));
   }
  }

  white_noise_.resize((num_spins * 3) * (2*num_trunc_ + 1));

  CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), white_noise_.device_data(), white_noise_.size(), 0.0, 1.0));

  #ifdef PRINT_NOISE
  debug_file_.open("noise.tsv");
  #endif

  #ifdef PRINT_MEMORY
  memory_file_.open("memory.tsv");
  #endif

}

void CudaLangevinArbitraryThermostat::update() {

  //
  // update filter if temperature changes (or is first iteration)
  //
  if (filter_temperature_ != this->temperature() || solver->iteration() == 0) {
    filter_temperature_ = this->temperature();

    vector<double> discrete_filter;
    if (do_zero_point_) {
      discrete_filter = discrete_psd_filter(
          std::function<double(double&, double&, double&, double&, double&)>(lorentzian_correlator),
          delta_omega_, num_freq_, delta_t_, temperature_, lorentzian_gamma_, lorentzian_omega0_, lorentzian_A_);
    } else {
      discrete_filter = discrete_psd_filter(
          std::function<double(double&, double&)>(barker_correlator),
          delta_omega_, num_freq_, delta_t_, temperature_);
    }

    std::ofstream noise_target_file(jams::output::full_path_filename("noise_target_spectrum.tsv"));
    for (auto i = 0; i < discrete_filter.size(); ++i) {
      noise_target_file << i << " " << i*delta_omega_/(kTwoPi) << " " << discrete_filter[i] << "\n";
    }
    noise_target_file.close();


    auto convoluted_filter = real_discrete_ft(discrete_filter);
//    filter_.resize(convoluted_filter.size());
//    std::copy(convoluted_filter.begin(), convoluted_filter.end(), filter_.begin());

    std::ofstream filter_full_file(jams::output::full_path_filename("noise_filter_full.tsv"));
    for (auto i = 0; i < convoluted_filter.size(); ++i) {
      filter_full_file << i*delta_t_ << " " << convoluted_filter[i] << "\n";
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

  //
  // Generate Noise Process
  //
  {
    assert(filter_.size() != 0);

    int block_size = 128;
    int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

    const double temperature = this->temperature();

    CHECK_CURAND_STATUS(
        curandSetStream(jams::instance().curand_generator(),
                        dev_curand_stream_));

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
    cuda_array_elementwise_scale(globals::num_spins, 3, sigma_.device_data(),
                                 sqrt(temperature), noise_.device_data(), 1,
                                 noise_.device_data(), 1, dev_stream_);


    #ifdef PRINT_NOISE
    debug_file_ << solver->iteration() * delta_t_ << " " << noise_(0, 0)
                << "\n";
    #endif
  }

  // generate new random numbers
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(jams::instance().curand_generator(),
                                 white_noise_.device_data() +
                                 globals::num_spins3 *
                                 pbc(solver->iteration() + num_trunc_,
                                     2 * num_trunc_ + 1),
                                 globals::num_spins3, 0.0, 1.0));

  //
  // Calculate memory contribution
  //
  {
    const dim3 block_size = {64, 3, 1};
    auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 3, 1});

    lorentzian_memory_cuda_kernel<<<grid_size,block_size, 0, dev_stream_>>>(
            memory_w_process_.device_data(),
            memory_v_process_.device_data(),
            globals::s.device_data(),
            globals::gyro.device_data(),
            lorentzian_omega0_,
            lorentzian_gamma_,
            lorentzian_A_,
            solver->time_step(),
            globals::num_spins);
  }

  #ifdef PRINT_MEMORY
  memory_file_ << solver->iteration() * delta_t_ << " " << memory_v_process_(0) << " " << memory_v_process_(1) << " " << memory_v_process_(2) << "\n";
  #endif

  // Sum noise and memory
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(),noise_.elements(), &kOne, memory_v_process_.data(), 1, noise_.device_data(), 1));
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