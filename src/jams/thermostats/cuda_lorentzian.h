#ifndef JAMS_CUDA_LORENTZIAN_THERMOSTAT_H
#define JAMS_CUDA_LORENTZIAN_THERMOSTAT_H

/// ==============================
/// Class CudaLorentzianThermostat
/// ==============================
/// Implements a thermostat based on harmonic oscillators with a Lorentzian
/// coupling function following Anders, arXiv:2009.00600
/// https://arxiv.org/abs/2009.00600.
///
/// The Lorentzian has two explicit parameters \omega_0 and \Gamma which
/// describe the center and width. The amplitude is linked to the material
/// damping parameter `alpha` through A = \alpha \omega_0^4 / \Gamma.
///
/// -----------
/// Limitations
/// -----------
///
/// This thermostat should always be used with the solver
/// "ll-lorentzian-rk4-gpu" (cuda_ll_lorentzian_rk4.cu). The thermostat is the
/// **fluctuation** part and the solver contains the **dissipation** part
/// (memory) of the fluctuation dissipation theorem.
///
/// The class should only be used with a single species because \alpha is
/// assumed to be the same for all spins.
///
/// ---------------
/// Config Settings
/// ---------------
///
/// * lorentzian_omega0 - float - required:
///     Center of the Lorentzian in THz.
///
/// * lorentzian_gamma - float - required:
///     Width of the Lorentzian in THz.
///
/// * spectrum - string - optional (default "quantum-lorentzian"):
///     Shape of the spectral function. Possible values are : "classical",
///     "classical-lorentzian", "quantum-lorentzian". "classical" exists
///     only for testing purposes.
///
///    "classical"            -> P(\omega) = 2 \alpha k_B T
///
///    "classical-lorentzian" -> P(\omega) = 2 k_B T (A \Gamma)
///                          / ((\omega_0^2 - \omega^2)^2 + (\omega\Gamma)^2)
///
///    "quantum-lorentzian"   -> P(\omega) = 2 k_B T \hbar |\omega|
///                          * coth((2 k_B T) / (\hbar |\omega|)) * (A \Gamma)
///                          / ((\omega_0^2 - \omega^2)^2 + (\omega\Gamma)^2)
///
/// * num_freq - int - optional (default 10000):
///     Number of frequencies used in Fourier method.
///
///     The maximum noise frequency we can represent is ω_max = π/Δt. The
///     frequency resolution will be ω_max / num_freq. We need sufficient
///     resolution such that any features (such as the Lorentzian) in the noise
///     PSD are well sampled. But we will be doing a real time convolution in
///     the end with a smaller number of points than num_freq.
///
/// * num_trunc - int - optional (default 2000):
///     Number of time points to use in the real time convolution. This must
///     best less than num_freq.
///
///     The real time kernel should decay to zero quite quickly, with a
///     dependence on the damping (Lorentzian width). So we can use fewer time
///     points in the real time convolution than frequency samples. This
///     parameter determines the memory requirement of the method which is
///     dominated by the number of random samples we must store which is
///     (2 * num_trunc + 1) * num_spins * 3.
///
/// --------------
/// Example Config
/// --------------
///
/// \verbatim
///
/// solver : {
///  module = "ll-lorentzian-rk4-gpu";
///  thermostat = "langevin-lorentzian-gpu";
///  t_step = 10e-16;
///  t_max  = 1e-9;
/// };
///
/// thermostat : {
///  spectrum = "classical-lorentzian";
///  lorentzian_omega0 = 3000.0;
///  lorentzian_gamma = 100;
///  num_freq = 100000;
///  num_trunc = 200;
/// }
///
/// \endverbatim
///
/// ------------
/// Output Files
/// ------------
///
/// <seed>_noise_target_spectrum.tsv:
/// ---------------------------------
///   The analytic function of the spectrum from which the convolution
///   filter is build (i.e. taking into account the maximum frequency and number
///   of discrete frequencies).
///
///   columns:
///     0: freq_THz     - frequency in THz
///     1: spectrum_meV - analytic spectrum in meV
///
/// <seed>_noise_filter_full.tsv:
/// -----------------------------
///   Contains the data of the convolution filter which will be applied to the
///   white noise to colour it. This is the full data set. In the actual
///   calculation we use a truncated sum of only the first `num_trunc` elements.
///
///   columns:
///     0: delta_t_ps   - time difference in ps
///     1: filter_arb   - value of the convolution filter at this time difference
///
/// <seed>_noise_filter_trunc.tsv:
/// -----------------------------
///   Contains the data of the truncated convolution filter, i.e. only the first
///   `num_trunc` elements of the full filter function.
///
///   columns:
///     0: delta_t_ps   - time difference in ps
///     1: filter_arb   - value of the convolution filter at this time difference

#if HAS_CUDA

#include <curand.h>
#include <fstream>
#include <mutex>

#include "jams/core/thermostat.h"

class CudaLorentzianThermostat : public Thermostat {
public:
    CudaLorentzianThermostat(const double &temperature, const double &sigma, const int num_spins);
    ~CudaLorentzianThermostat();

    void update();

    // override the base class implementation
    const double* device_data() { return noise_.device_data(); }

private:

    void output_thermostat_properties(std::ostream& os);


    // Typedef for the signature of the spectral function. Mathematicall
    // this is something like f(omega, ...). The first argument will always be
    // omega (the frequencies over which the spectral function is defined and
    // then any additional parameters can be passed after this.
    template<typename... Args>
    using SpectralFunctionSignature = std::function<double(double, Args...)>;

    // Returns a vector containing sqrt{f(i * delta_omega, ...)}, i.e. the
    // spectral function f(omega, ...) sampled at regular discrete points
    // delta_omega apart.
    template<typename... Args>
    std::vector<double> discrete_psd_filter(
        SpectralFunctionSignature<Args...> spectral_function
        , const double& delta_omega
        , const int& num_freq
        , Args&&... args);

    // Returns the 1D discrete fourier transform of x. This is a properly
    // normalised transform.
    std::vector<double> discrete_real_fourier_transform(std::vector<double>& x);


    std::ofstream debug_file_;

    int    num_freq_;
    int    num_trunc_;
    double max_omega_;
    double delta_omega_;
    double delta_t_;
    double filter_temperature_;

    // 3*num_spins increased by one to make it even if needed. Required for
    // curand which can only generate even lengths of random numbers.
    int    num_spins3_even_;

    // Lorentzian parameters
    double lorentzian_omega0_;
    double lorentzian_gamma_;
    double lorentzian_A_;

    jams::MultiArray<double, 1> filter_;
    jams::MultiArray<double, 1> white_noise_;

    cudaStream_t                dev_stream_ = nullptr;
    cudaStream_t                dev_curand_stream_ = nullptr;
};

// INLINE DEFINITIONS

template<typename... Args>
std::vector<double> CudaLorentzianThermostat::discrete_psd_filter(
    SpectralFunctionSignature<Args...> spectral_function,
    const double &delta_omega, const int &num_freq, Args &&... args) {

  std::vector<double> result(num_freq);

  for(auto i = 0; i < result.size(); ++i) {
    double omega = i * delta_omega;
    result[i] = sqrt(spectral_function(omega, args...));
  }

  return result;
}

#endif  // CUDA
#endif  // JAMS_CUDA_LORENTZIAN_THERMOSTAT_H
