#ifndef JAMS_CUDA_LORENTZIAN_THERMOSTAT_H
#define JAMS_CUDA_LORENTZIAN_THERMOSTAT_H

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

    bool   use_classical_noise_;
    int    num_freq_;
    int    num_trunc_;
    double max_omega_;
    double delta_omega_;
    double delta_t_;
    double filter_temperature_;

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
