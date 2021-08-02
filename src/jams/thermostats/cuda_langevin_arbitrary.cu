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

double barker_correlator(double& omega, double& temperature) {
  if (omega == 0.0) return 1.0;
  double x = (kHBarIU * abs(omega)) / (kBoltzmannIU * temperature);
  return x / (exp(x) - 1);
}


}

CudaLorentzianThermostat::CudaLorentzianThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  filter_temperature_(0.0) {

  if (lattice->num_materials() > 1) {
    throw std::runtime_error("CudaLangevinArbitraryThermostat is only implemented for single material cells");
  }
  cout << "\n  initialising CUDA Langevin arbitrary noise thermostat\n";


  // number of frequencies to discretize the spectrum over
  num_freq_ = jams::config_optional(config->lookup("thermostat"), "num_freq", 10000);
  // number of terms to use in the convolution sum
  num_trunc_ = jams::config_optional(config->lookup("thermostat"), "num_trunc", 2000);
  assert(num_trunc_ <= num_freq_);

  lorentzian_gamma_ = jams::config_required<double>(config->lookup("thermostat"), "lorentzian_gamma"); // 3.71e12 * kTwoPi;
  lorentzian_omega0_ = jams::config_required<double>(config->lookup("thermostat"), "lorentzian_omega0"); // 6.27e12 * kTwoPi;
  lorentzian_A_ =  (globals::alpha(0) * pow4(lorentzian_omega0_)) / (lorentzian_gamma_);

  delta_t_ = solver->time_step();
  max_omega_ = kPi / delta_t_;
  delta_omega_ = max_omega_ / double(num_freq_);

  output_thermostat_properties(std::cout);


  if (cudaStreamCreate(&dev_stream_) != cudaSuccess) {
   jams_die("Failed to create CUDA stream in CudaLangevinArbitraryThermostat");
  }

  if (cudaStreamCreate(&dev_curand_stream_) != cudaSuccess) {
   jams_die("Failed to create CURAND stream in CudaLangevinArbitraryThermostat");
  }

  // If the noise term is written as a B-field in Tesla it should look like:
  //
  // B(t) = \sigma_i \xi(t, T)
  //
  // where the \sigma_i depends on parameters on the local site and \xi
  // is a dimensionless noise.
  //
  // \sigma_i = \sqrt((2 \alpha_i k_B T) / (\gamma_i \mu_{s,i}))
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

  // We store the temperature and check that it doesn't change when running
  // the simulation. Currently we don't know how to deal with a dynamically
  // changing temperature in this formalism.
  filter_temperature_ = this->temperature();

  for (int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      sigma_(i, j) = sqrt(1.0 /solver->time_step());
    }
  }

  // Do the initial population of white noise
  white_noise_.resize((num_spins * 3) * (2*num_trunc_ + 1));
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(jams::instance().curand_generator(),
                                 white_noise_.device_data(), white_noise_.size(), 0.0, 1.0));

  // Define the spectral function P(omega) as a lambda
  std::function<double(double&)> lorentzian_spectral_function = [&](double& omega) {
    if (omega == 0.0) return 1.0;

    double lorentzian = (lorentzian_A_ * lorentzian_gamma_ * kHBarIU * abs(omega))
        / (pow2(pow2(lorentzian_omega0_) - pow2(omega)) + pow2(omega * lorentzian_gamma_));

    double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * filter_temperature_);
    return lorentzian * coth(x);
  };

  // Generate the discrete filter F(n) = sqrt(P(n * delta_omega))
  auto discrete_filter = discrete_psd_filter(lorentzian_spectral_function, delta_omega_, num_freq_);

  // Do the fourier transform to go from frequency space into sqrt(P(omega)) -> K(t - t')
  auto convoluted_filter = discrete_real_fourier_transform(discrete_filter);

  // We will do a truncated sum for the real time convolution so we only need to copy
  // forward part of the filter.
  filter_.resize(num_trunc_);
  std::copy(convoluted_filter.begin(), convoluted_filter.begin()+num_trunc_, filter_.begin());

  // Output functions to files for checking
  std::ofstream noise_target_file(jams::output::full_path_filename("noise_target_spectrum.tsv"));
  for (auto i = 0; i < discrete_filter.size(); ++i) {
    noise_target_file << i << " " << i*delta_omega_/(kTwoPi) << " " << discrete_filter[i] << "\n";
  }
  noise_target_file.close();


  std::ofstream filter_full_file(jams::output::full_path_filename("noise_filter_full.tsv"));
  for (auto i = 0; i < convoluted_filter.size(); ++i) {
    filter_full_file << i*delta_t_ << " " << convoluted_filter[i] << "\n";
  }
  filter_full_file.close();


  std::ofstream filter_file(jams::output::full_path_filename("noise_filter_trunc.tsv"));
  for (auto i = 0; i < filter_.size(); ++i) {
    filter_file << i << " " << filter_(i) << "\n";
  }
  filter_file.close();

  #ifdef PRINT_NOISE
  debug_file_.open("noise.tsv");
  #endif
}

void CudaLorentzianThermostat::update() {
  assert(filter_.size() != 0);

  if (filter_temperature_ != this->temperature()) {
    throw runtime_error(
        "you cannot dynamically change the temperature for the CudaLangevinArbitraryThermostat");
  }

  int block_size = 128;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  CHECK_CURAND_STATUS(
      curandSetStream(jams::instance().curand_generator(), dev_curand_stream_));

  const auto n = pbc(solver->iteration(), (2 * num_trunc_ + 1));
  arbitrary_stochastic_process_cuda_kernel<<<grid_size, block_size, 0, dev_stream_ >>>(
      noise_.device_data(),
      filter_.device_data(),
      white_noise_.device_data(),
      n,
      num_trunc_,
      globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  // scale the noise by the prefactor sigma
  cuda_array_elementwise_scale(globals::num_spins, 3, sigma_.device_data(), 1.0, noise_.device_data(), 1,
                               noise_.device_data(), 1, dev_stream_);

  // generate new random numbers ready for the next round
  auto start_index = globals::num_spins3 * pbc(solver->iteration() + num_trunc_, 2 * num_trunc_ + 1);
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(
          jams::instance().curand_generator(),
           white_noise_.device_data() + start_index, globals::num_spins3, 0.0, 1.0));

  #ifdef PRINT_NOISE
  debug_file_ << solver->iteration() * delta_t_ << " " << noise_(0, 0)
                << "\n";
  #endif

  // reset the curand stream to default
  CHECK_CURAND_STATUS(
      curandSetStream(jams::instance().curand_generator(), nullptr));
}

CudaLorentzianThermostat::~CudaLorentzianThermostat() {
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


void CudaLorentzianThermostat::output_thermostat_properties(std::ostream& os) {
  os << "    lorentzian gamma (THz) " << std::fixed << lorentzian_gamma_ / (kTwoPi) << "\n";
  os << "    lorentzian omega0 (THz) " << std::fixed << lorentzian_omega0_ / (kTwoPi) << "\n";
  os << "    lorentzian A " << std::fixed << lorentzian_A_ << "\n";

  os << "    max_omega (THz) " << std::fixed << max_omega_ / (kTwoPi) << "\n";
  os << "    delta_omega (THz) " << std::fixed << delta_omega_ / (kTwoPi) << "\n";
  os << "    delta_t " << std::scientific << delta_t_ << "\n";
  os << "    num_freq " << num_freq_ << "\n";
  os << "    num_trunc " << num_trunc_ << "\n";
}


std::vector<double> CudaLorentzianThermostat::discrete_real_fourier_transform(std::vector<double>& x) {
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


#undef PRINT_NOISE