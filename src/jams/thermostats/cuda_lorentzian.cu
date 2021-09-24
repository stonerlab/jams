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
#include "jams/thermostats/cuda_lorentzian.h"
#include "jams/thermostats/cuda_langevin_arbitrary_kernel.h"
#include "jams/interface/fft.h"
#include "jams/maths/functions.h"

//#define PRINT_NOISE

CudaLorentzianThermostat::CudaLorentzianThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  filter_temperature_(0.0) {

  std::cout << "\n  initialising CUDA Langevin arbitrary noise thermostat\n";

  if (lattice->num_materials() > 1) {
    throw std::runtime_error(
        "CudaLangevinArbitraryThermostat is only implemented for single material cells");
  }

  auto noise_spectrum_type = lowercase(
      jams::config_optional<std::string>(config->lookup("thermostat"),
                                         "spectrum", "quantum-lorentzian"));

  // number of frequencies to discretize the spectrum over
  num_freq_ = jams::config_optional(config->lookup("thermostat"), "num_freq",
                                    10000);
  // number of terms to use in the convolution sum
  num_trunc_ = jams::config_optional(config->lookup("thermostat"), "num_trunc",
                                     2000);
  assert(num_trunc_ <= num_freq_);

  lorentzian_gamma_ = kTwoPi * jams::config_required<double>(
      config->lookup("thermostat"), "lorentzian_gamma");
  lorentzian_omega0_ = kTwoPi * jams::config_required<double>(
      config->lookup("thermostat"), "lorentzian_omega0");


  delta_t_ = solver->time_step();
  max_omega_ = kPi / delta_t_;
  delta_omega_ = max_omega_ / double(num_freq_);

  // In arXiv:2009.00600v2 Janet uses eta_G for the Gilbert damping, but this is
  // a **dimensionful** Gilbert damping (implied by Eq. (1) in the paper and
  // also explicitly mentioned). In JAMS alpha is the dimensionless Gilbert
  // damping. The difference is eta_G = alpha / (mu_s * gamma). It's important
  // that we convert here to get the scaling of the noice correct (i.e. it
  // converts Janet's equations into the JAMS convention).

  double eta_G = globals::alpha(0) / (globals::mus(0) * globals::gyro(0));

  lorentzian_A_ =
      (eta_G * pow4(lorentzian_omega0_)) / (lorentzian_gamma_);

  output_thermostat_properties(std::cout);


  if (cudaStreamCreate(&dev_stream_) != cudaSuccess) {
    jams_die("Failed to create CUDA stream in CudaLangevinArbitraryThermostat");
  }

  if (cudaStreamCreate(&dev_curand_stream_) != cudaSuccess) {
    jams_die(
        "Failed to create CURAND stream in CudaLangevinArbitraryThermostat");
  }

  // We store the temperature and check in update() that it doesn't change when
  // running the simulation. Currently we don't know how to deal with a
  // dynamically changing temperature in this formalism.
  filter_temperature_ = this->temperature();

  for (int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      sigma_(i, j) = sqrt(1.0 / solver->time_step());
    }
  }

  // Curand will only generate noise for arrays which are multiples of 2
  // so we sometimes have to make the size of the white noise array artificially
  // larger. We can't just add 1 element to the total white_noise_ because when
  // we generate random numbers later we will be generating only 3*num_spins
  // numbers at each step. So we adjust 3*num_spins to make sure it's even.

  num_spins3_even_ = (3 * num_spins) + ((3 * num_spins) % 2);

  white_noise_.resize(num_spins3_even_ * (2 * num_trunc_ + 1));
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(jams::instance().curand_generator(),
                                 white_noise_.device_data(),
                                 white_noise_.size(), 0.0, 1.0));


  // Define the spectral function P(omega) as a lambda
  std::function<double(double)> lorentzian_spectral_function;

  if (noise_spectrum_type == "classical") {
    lorentzian_spectral_function = [&](double omega) {
        return 2.0 * kBoltzmannIU * filter_temperature_ * eta_G;
    };
  } else if (noise_spectrum_type == "classical-lorentzian") {
    lorentzian_spectral_function = [&](double omega) {
        double lorentzian = (lorentzian_A_ * lorentzian_gamma_ * kHBarIU * abs(omega))
                            / (pow2(pow2(lorentzian_omega0_) - pow2(omega)) + pow2(omega * lorentzian_gamma_));

        double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * filter_temperature_);

        // Need to avoid undefined calculations (1/0 and coth(0)) so here we use
        // the analytic limits for omega == 0.0
        if (omega == 0.0) {
          return (2.0 * kBoltzmannIU * filter_temperature_) *
                 (lorentzian_A_ * lorentzian_gamma_) / pow4(lorentzian_omega0_);
        }
        return lorentzian / x;
    };
  } else if (noise_spectrum_type == "quantum-lorentzian") {
    lorentzian_spectral_function = [&](double omega) {
        double lorentzian = (lorentzian_A_ * lorentzian_gamma_ * kHBarIU * abs(omega))
                            / (pow2(pow2(lorentzian_omega0_) - pow2(omega)) + pow2(omega * lorentzian_gamma_));

        double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * filter_temperature_);

        // Need to avoid undefined calculations (1/0 and coth(0)) so here we use
        // the analytic limits for omega == 0.0
        if (omega == 0.0) {
          return (2.0 * kBoltzmannIU * filter_temperature_) *
                 (lorentzian_A_ * lorentzian_gamma_) / pow4(lorentzian_omega0_);
        }
        return lorentzian * jams::maths::coth(x);
    };
  }
  else {
    throw std::runtime_error("unknown spectrum type '" + noise_spectrum_type +"'");
  }

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
  noise_target_file << "freq_THz    spectrum_meV" << std::endl;
  for (auto i = 0; i < discrete_filter.size(); ++i) {
    noise_target_file << jams::fmt::decimal << i*delta_omega_/(kTwoPi) << " ";
    noise_target_file << jams::fmt::sci     << lorentzian_spectral_function(i*delta_omega_) << "\n";
  }
  noise_target_file.close();

  std::ofstream filter_full_file(jams::output::full_path_filename("noise_filter_full.tsv"));
  filter_full_file << "delta_t_ps    filter_arb" << std::endl;
  for (auto i = 0; i < convoluted_filter.size(); ++i) {
    filter_full_file << jams::fmt::decimal << i*delta_t_ << " ";
    filter_full_file << jams::fmt::sci     << convoluted_filter[i] << "\n";
  }
  filter_full_file.close();

  std::ofstream filter_file(jams::output::full_path_filename("noise_filter_trunc.tsv"));
  filter_file << "delta_t_ps    filter_arb" << std::endl;
  for (auto i = 0; i < filter_.size(); ++i) {
    filter_file << jams::fmt::decimal << i*delta_t_ << " ";
    filter_file << jams::fmt::sci      << filter_(i) << "\n";
  }
  filter_file.close();

  #ifdef PRINT_NOISE
  debug_file_.open("noise.tsv");
  #endif
}

void CudaLorentzianThermostat::update() {
  assert(filter_.size() != 0);

  if (filter_temperature_ != this->temperature()) {
    throw std::runtime_error(
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
  auto start_index = num_spins3_even_ * pbc(solver->iteration() + num_trunc_, 2 * num_trunc_ + 1);
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(
          jams::instance().curand_generator(),
           white_noise_.device_data() + start_index, num_spins3_even_, 0.0, 1.0));

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
      FFTW_REDFT10, // fftw_r2r_kind kind
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