// Copyright 2014 Joseph Barker. All rights reserved.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <jams/common.h>

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
#include "jams/helpers/finite.h"
#include "jams/helpers/random.h"
#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_common.h"
#include "jams/monitors/magnetisation.h"
#include "jams/thermostats/cuda_thermostat_general_fft.h"
#include "jams/thermostats/cuda_thermostat_general_fft_kernel.h"
#include "jams/interface/fft.h"
#include "jams/maths/functions.h"
#include <jams/helpers/exception.h>
#include <jams/helpers/mixed_precision.h>

//#define PRINT_NOISE

CudaThermostatGeneralFFT::CudaThermostatGeneralFFT(
    const jams::Real &temperature, const std::span<const double> coupling,
    const double timestep, const int num_spins, const std::string& noise_spectrum_type,
    const std::optional<jams::LorentzianBathParameters> memory_bath)
: Thermostat(temperature, 0.0, static_cast<jams::Real>(timestep), num_spins),
  delta_t_(timestep), filter_temperature_(this->temperature()), site_scale_(num_spins) {

  std::cout << "\n  initialising general-fft-gpu thermostat\n";

  if (!jams::is_finite(timestep) || timestep <= 0.0) {
    throw std::runtime_error("general-fft-gpu requires a positive finite sampling interval");
  }

  if (has_per_spin_temperature()) {
    throw std::runtime_error(
        "general-fft-gpu thermostat does not support per-spin temperature profiles");
  }

  if (globals::lattice->num_materials() > 1) {
    throw std::runtime_error(
        "CudaLangevinArbitraryThermostat is only implemented for single material cells");
  }

  // All sites share the spectrum's shape. Build the filter at the largest
  // coupling and scale each field by the square root of its relative power.
  const double eta_G = coupling.empty() ? 0.0 : *std::max_element(coupling.begin(), coupling.end());
  for (int i = 0; i < num_spins; ++i) {
    site_scale_(i) = static_cast<jams::Real>(
        eta_G > 0.0 ? std::sqrt((coupling[i] / eta_G) / timestep) : 0.0);
  }

  if (noise_spectrum_type.ends_with("-lorentzian")) {
    const auto parameters = memory_bath ? *memory_bath :
        jams::LorentzianBathParameters::from_settings(globals::config->lookup("thermostat"));
    lorentzian_omega0_ = parameters.omega0;
    lorentzian_gamma_ = parameters.linewidth;
    lorentzian_A_ = parameters.amplitude(eta_G);
  }

  // Define the spectral function P(omega) as a lambda
  std::function<double(double)> psd_function;

  if (noise_spectrum_type == "classical-ohmic" || noise_spectrum_type == "classical") {
    psd_function = [&](double omega) {
      return classical_ohmic_spectrum(omega, filter_temperature_, eta_G);
    };
  } else if (noise_spectrum_type == "quantum-ohmic") {
    psd_function = [&](double omega) {
      return quantum_ohmic_spectrum(omega, filter_temperature_, eta_G);
    };
  } else if (noise_spectrum_type == "quantum-no-zero-ohmic" || noise_spectrum_type == "bose-einstein") {
    psd_function = [&](double omega) {
      return quantum_no_zero_ohmic_spectrum(omega, filter_temperature_, eta_G);
    };
  } else if (noise_spectrum_type == "classical-lorentzian") {
    psd_function = [&](double omega) {
      return classical_lorentzian_spectrum(omega, filter_temperature_, lorentzian_omega0_, lorentzian_gamma_, lorentzian_A_);
    };
  } else if (noise_spectrum_type == "quantum-lorentzian") {
    psd_function = [&](double omega) {
      return quantum_lorentzian_spectrum(omega, filter_temperature_, lorentzian_omega0_, lorentzian_gamma_, lorentzian_A_);
    };
  } else if (noise_spectrum_type == "quantum-no-zero-lorentzian" || noise_spectrum_type == "no-zero-quantum-lorentzian") {
    psd_function = [&](double omega) {
      return quantum_no_zero_lorentzian_spectrum(omega, filter_temperature_, lorentzian_omega0_, lorentzian_gamma_, lorentzian_A_);
    };
  } else {
    throw std::runtime_error("unknown spectrum type '" + noise_spectrum_type +"'");
  }

  // Autoconfigure the frequency resolution for the PSD omega -> t transform.
  //
  // If the resolution is too low then we sample features such as the Lorentzian too crudely which means the memory
  // kernel is a poor approximation of the analytic Fourier transform. To calculate an appropriate resolution we
  // successively increase the resolution until the memory kernel stops changing with increasing resolution. Note that
  // the frequency resolution usually needs to be much higher than the final time range of the memory.

  const int num_freq_init = 10000;
  const int num_freq_increment = 10000;
  const int num_freq_increment_attempts = 499;
  const double memory_function_equality_tolerance = 1e-7;

  num_freq_ = num_freq_init;
  max_omega_ = kPi / delta_t_;
  delta_omega_ = max_omega_ / double(num_freq_);

  // Generate the discrete filter F(n) = sqrt(P(n * delta_omega))
  auto discrete_psd = discrete_sqrt_psd(psd_function, delta_omega_, num_freq_);
  // Do the fourier transform to go from frequency space into sqrt(P(omega)) -> K(t - t')
  auto memory_kernel = discrete_real_fourier_transform(discrete_psd);

  for (auto n = 0; n < num_freq_increment_attempts; ++n) {
    const int trial_num_freq = num_freq_ + num_freq_increment;
    const double trial_delta_omega = max_omega_ / double(trial_num_freq);
    auto trial_psd = discrete_sqrt_psd(psd_function, trial_delta_omega, trial_num_freq);
    auto trial_kernel = discrete_real_fourier_transform(trial_psd);

    bool kernels_are_equal = std::equal(memory_kernel.begin(), memory_kernel.end(), trial_kernel.begin(),
                                        [&](const auto& x, const auto &y){
                 return approximately_equal(x, y, memory_function_equality_tolerance);
    });

    if (kernels_are_equal) {
      break;
    }

    num_freq_ = trial_num_freq;
    delta_omega_ = trial_delta_omega;
    discrete_psd = std::move(trial_psd);
    memory_kernel = std::move(trial_kernel);
  }


  // Autoconfigure the number of memory kernel time steps to retain
  //
  // The Fourier transform of the PSD requires a high frequency resolution, but the memory kernel itself is usually
  // fairly short ranged in time. So we only need to keep a truncated part of the kernel. We check how much to keep by
  // calculating the mean of successive blocks of the kernel and stopping once we have two successive blocks with
  // means close to zero.

  const double memory_absmax = std::abs(*std::max_element(memory_kernel.begin(), memory_kernel.end(),
    [](const auto& a, const auto& b){
    return std::abs(a) < std::abs(b);
  }));

  const int trunc_block_size = 100;
  const double trunc_zero_tolerance = memory_absmax * 0.005;

  num_trunc_ = static_cast<int>(memory_kernel.size());
  if (memory_absmax == 0.0) {
    num_trunc_ = 1;
  } else {
    bool previous_block_near_zero = false;
    for (std::size_t i = 0; i < memory_kernel.size(); i += trunc_block_size) {
      const auto end = std::min(i + trunc_block_size, memory_kernel.size());
      const double block_sum = std::accumulate(memory_kernel.begin() + i,
                                               memory_kernel.begin() + end, 0.0);
      const bool block_near_zero = approximately_zero(block_sum / double(end - i),
                                                       trunc_zero_tolerance);
      if (previous_block_near_zero && block_near_zero) {
        num_trunc_ = static_cast<int>(i);
        break;
      }
      previous_block_near_zero = block_near_zero;
    }
  }

  memory_kernel_.resize(num_trunc_);
  std::copy(memory_kernel.begin(), memory_kernel.begin()+num_trunc_, memory_kernel_.begin());

  if (cudaStreamCreate(&dev_stream_) != cudaSuccess) {
    throw jams::GeneralException("Failed to create CUDA stream in CudaLangevinBoseThermostat");
  }

  // Curand will only generate noise for arrays which are multiples of 2
  // so pad each ring row. The output still has exactly 3*num_spins entries.

  const auto num_proc = 3 * static_cast<std::size_t>(num_spins);
  num_spins3_even_ = num_proc + (num_proc % 2);

  white_noise_.resize(num_spins3_even_ * (2 * num_trunc_ + 1));
  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_stream_));
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(jams::instance().curand_generator(),
                                 white_noise_.mutable_device_data(),
                                 white_noise_.size(), 0.0, 1.0));
  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), nullptr));


  // Output functions to files for checking
  std::ofstream noise_target_file(jams::output::full_path_filename("noise_target_spectrum.tsv"));
  noise_target_file << "freq_THz    spectrum_meV" << std::endl;
  for (auto i = 0; i < discrete_psd.size(); ++i) {
    noise_target_file << jams::fmt::decimal << i*delta_omega_/(kTwoPi) << " ";
    noise_target_file << jams::fmt::sci << pow2(discrete_psd[i]) << "\n";
  }
  noise_target_file.close();

  std::ofstream filter_full_file(jams::output::full_path_filename("noise_filter_full.tsv"));
  filter_full_file << "delta_t_ps    filter_arb" << std::endl;
  for (auto i = 0; i < memory_kernel.size(); ++i) {
    filter_full_file << jams::fmt::decimal << i*delta_t_ << " ";
    filter_full_file << jams::fmt::sci << memory_kernel[i] << "\n";
  }
  filter_full_file.close();

  std::ofstream filter_file(jams::output::full_path_filename("noise_filter_trunc.tsv"));
  filter_file << "delta_t_ps    filter_arb" << std::endl;
  for (auto i = 0; i < memory_kernel_.size(); ++i) {
    filter_file << jams::fmt::decimal << i*delta_t_ << " ";
    filter_file << jams::fmt::sci << memory_kernel_(i) << "\n";
  }
  filter_file.close();

  #ifdef PRINT_NOISE
  debug_file_.open("noise.tsv");
  #endif

  output_thermostat_properties(std::cout);
}

void CudaThermostatGeneralFFT::update() {
  assert(memory_kernel_.size() != 0);

  if (filter_temperature_ != this->temperature()) {
    throw std::runtime_error(
        "you cannot dynamically change the temperature for the CudaLangevinArbitraryThermostat");
  }

  const int block_size = 128;
  const auto num_proc = static_cast<int>(noise_.elements());
  const auto grid_size = (noise_.elements() + block_size - 1) / block_size;

  const auto n = static_cast<int>(noise_sample_count_ % (2 * num_trunc_ + 1));
  cuda_thermostat_general_fft_kernel<<<grid_size, block_size, 0, dev_stream_ >>>(
      noise_.mutable_device_data(),
      memory_kernel_.device_data(),
      white_noise_.device_data(),
      n,
      num_trunc_,
      num_proc,
      num_spins3_even_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  // scale the noise by the prefactor sigma
  cuda_array_elementwise_scale(num_proc / 3, 3, site_scale_.device_data(), 1.0, noise_.mutable_device_data(), 1,
                               noise_.mutable_device_data(), 1, dev_stream_);

  // Order the refill and convolution on the same stream so that consecutive
  // updates can safely reuse the ring buffer.
  auto start_index = num_spins3_even_ * pbc(
      n + num_trunc_ + 1, 2 * num_trunc_ + 1);
  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_stream_));
  CHECK_CURAND_STATUS(
      curandGenerateNormalDouble(
          jams::instance().curand_generator(),
           white_noise_.mutable_device_data() + start_index, num_spins3_even_, 0.0, 1.0));

  #ifdef PRINT_NOISE
  debug_file_ << noise_sample_count_ * delta_t_ << " " << noise_(0, 0)
                << "\n";
  #endif

  // reset the curand stream to default
  CHECK_CURAND_STATUS(
      curandSetStream(jams::instance().curand_generator(), nullptr));
  ++noise_sample_count_;
}

CudaThermostatGeneralFFT::~CudaThermostatGeneralFFT() {
  #ifdef PRINT_NOISE
    debug_file_.close();
  #endif

  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }
}


void CudaThermostatGeneralFFT::output_thermostat_properties(std::ostream& os) {
  os << "    lorentzian gamma (THz) " << std::fixed << lorentzian_gamma_ / (kTwoPi) << "\n";
  os << "    lorentzian omega0 (THz) " << std::fixed << lorentzian_omega0_ / (kTwoPi) << "\n";
  os << "    lorentzian A " << std::fixed << lorentzian_A_ << "\n";

  os << "    max_omega (THz) " << std::fixed << max_omega_ / (kTwoPi) << "\n";
  os << "    delta_omega (THz) " << std::fixed << delta_omega_ / (kTwoPi) << "\n";
  os << "    delta_t " << std::scientific << delta_t_ << "\n";
  os << "    num_freq " << num_freq_ << "\n";
  os << "    num_trunc " << num_trunc_ << "\n";
}


std::vector<double> CudaThermostatGeneralFFT::discrete_real_fourier_transform(std::vector<double> x) {
  int size = static_cast<int>(x.size());
  fftw_plan plan = fftw_plan_r2r_1d(
      size, //int n
      x.data(), // double *in
      x.data(), // double *out
      FFTW_REDFT10, // fftw_r2r_kind kind
      FFTW_ESTIMATE); // unsigned flags

  if (plan == nullptr) {
    throw std::runtime_error("failed to create FFT thermostat filter transform");
  }

  fftw_execute(plan);
  fftw_destroy_plan(plan);

  // normalise by the logical size of the DFT
  // (see http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029)
  for (double &i : x) {
    i /= 2*size;
  }

  return x;
}

double CudaThermostatGeneralFFT::classical_ohmic_spectrum(double omega, double temperature, double eta_G) {
    return 2.0 * kBoltzmannIU * temperature * eta_G;
}

double CudaThermostatGeneralFFT::quantum_ohmic_spectrum(double omega, double temperature, double eta_G) {
  if (temperature == 0.0) {
    return kHBarIU * std::abs(omega) * eta_G;
  }
  if (omega == 0.0) {
    return 2.0 * kBoltzmannIU * temperature * eta_G;
  }

  double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return 2.0 * kBoltzmannIU * temperature * eta_G * x * jams::maths::coth(x);
}

double CudaThermostatGeneralFFT::quantum_no_zero_ohmic_spectrum(double omega, double temperature, double eta_G) {
  if (temperature == 0.0) {
    return 0.0;
  }
  if (omega == 0.0) {
    return 2.0 * kBoltzmannIU * temperature * eta_G;
  }

  double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return 2.0 * kBoltzmannIU * temperature * eta_G * x * (jams::maths::coth(x) - 1.0);
}


double CudaThermostatGeneralFFT::classical_lorentzian_spectrum(double omega, double temperature, double omega0, double gamma, double A) {
  // Need to avoid undefined calculations (1/0 and coth(0)) so here we use
  // the analytic limits for omega == 0.0
  if (omega == 0.0) {
    return (2.0 * kBoltzmannIU * temperature) * (A * gamma) / pow4(omega0);
  }

  double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * temperature);

  return kHBarIU * jams::maths::lorentzian(abs(omega), omega0, gamma, A) / x;
}


double CudaThermostatGeneralFFT::quantum_lorentzian_spectrum(double omega, double temperature, double omega0, double gamma, double A) {
  // Need to avoid undefined calculations (1/0 and coth(0)) so here we use
  // the analytic limits for omega == 0.0
  if (omega == 0.0) {
    return (2.0 * kBoltzmannIU * temperature) * (A * gamma) / pow4(omega0);
  }

  double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * temperature);

  return kHBarIU * jams::maths::lorentzian(abs(omega), omega0, gamma, A) * jams::maths::coth(x);
}


double CudaThermostatGeneralFFT::quantum_no_zero_lorentzian_spectrum(double omega, double temperature, double omega0, double gamma, double A) {
  // Need to avoid undefined calculations (1/0 and coth(0)) so here we use
  // the analytic limits for omega == 0.0
  if (omega == 0.0) {
    return (2.0 * kBoltzmannIU * temperature) * (A * gamma) / pow4(omega0);
  }

  double x = (kHBarIU * abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return kHBarIU * jams::maths::lorentzian(abs(omega), omega0, gamma, A) * (jams::maths::coth(x) - 1.0);
}

#undef PRINT_NOISE
