// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_THERMOSTAT_H
#define JAMS_CORE_THERMOSTAT_H

#include "jams/containers/multiarray.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "jams/helpers/mixed_precision.h"
#include "jams/core/thermostat_temperature_profile.h"

#if HAS_CUDA
#include "jams/cuda/cuda_stream.h"
#endif

namespace libconfig { class Setting; }

namespace jams {
enum class ThermalTorque { llg, additive, precession };
enum class ThermostatSamples { step_field, correlated_field, white_diffusion };

struct ThermostatStageRequest {
  enum class Mode { automatic, rs2 };
  double interval;
  Mode mode = Mode::automatic;
  bool changing_temperature = false;
};

inline bool spde_supports_stage_sampling(bool zero_point, bool per_spin_temperature,
                                         bool changing_temperature) {
  return !zero_point && (per_spin_temperature || !changing_temperature);
}

// Coefficient eta in the classical field spectrum 2 k_B T eta. Local
// equations supply gyro_eff. The retarded precession equation supplies
// bare gamma to set the strength of its memory kernel and matched noise.
std::vector<double> thermal_field_coupling(
    const MultiArray<Real, 1>& gyro, ThermalTorque torque = ThermalTorque::llg);

struct LorentzianBathParameters {
  double omega0;
  double linewidth;

  static LorentzianBathParameters from_settings(const libconfig::Setting& settings);
  double amplitude(double coupling) const;
};
}

class Thermostat {
 public:
  Thermostat(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins);

  virtual ~Thermostat();
  virtual void update() = 0;
  // Publish the starting sample before the caller advances the bath.
  virtual void sample_initial() { update(); }
  virtual jams::ThermostatSamples samples() const { return jams::ThermostatSamples::step_field; }
  // update() publishes independent Gaussian field components with variance
  // 2 k_B T eta / dt, where dt is the thermostat's sampling interval.
  virtual bool supplies_white_field() const { return false; }
  // Eligibility for automatic stage sampling is narrower than availability
  // of a correlated field. An existing solver may already use that field.
  virtual bool supports_stage_sampling(bool /*changing_temperature*/) const { return false; }
  virtual jams::MultiArray<double, 1> diffusion_amplitudes(double uniform_temperature) const;
  virtual bool supports_update_in_parallel() const { return false; }
  virtual void update_in_parallel();

  // factory
  // Coupling is set by the solver's thermal torque. White fields use
  // timestep as the sampling interval. RS2 uses native Wiener increments
  // for white noise and the separate stage interval for coloured noise.
  static Thermostat* create(const std::string &thermostat_name, double timestep,
                            std::span<const double> coupling,
                            std::optional<jams::ThermostatStageRequest> stage_request = std::nullopt,
                            std::optional<jams::LorentzianBathParameters> memory_bath = std::nullopt);

  // accessors
  jams::Real temperature() const { return temperature_; }
  void set_temperature(jams::Real T);

  bool has_per_spin_temperature() const { return temperature_profile_.is_per_spin(); }
  bool has_uniform_temperature() const { return temperature_profile_.is_uniform(); }
  const jams::ThermostatTemperatureProfile& temperature_profile() const { return temperature_profile_; }

  virtual const jams::Real* device_data() { return noise_.device_data(); }
  virtual const jams::Real* data() { return noise_.data(); }

  virtual jams::Real field(int i, int j) { return noise_(i, j); }


#if HAS_CUDA
  virtual cudaStream_t& get_stream()
  {
    return cuda_stream_.get();
  }

  // Call after enqueueing work to update the completion marker.
  void record_done()
  {
    cudaEventRecord(done_, get_stream());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  // Make an external stream wait for the thermostat's work.
  void wait_on(cudaStream_t external) const {
    cudaStreamWaitEvent(external, done_, 0);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  // Return ownership of the noise buffer after an external stream has
  // finished reading it. The next update waits for this marker before it may
  // overwrite noise_.
  void record_consumed(cudaStream_t external)
  {
    cudaEventRecord(consumed_, external);
    has_pending_consumer_ = true;
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  void wait_for_consumed()
  {
    if (!has_pending_consumer_) return;
    cudaStreamWaitEvent(get_stream(), consumed_, 0);
    has_pending_consumer_ = false;
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  // Host-side wait.
  void synchronize_done() const {
    cudaEventSynchronize(done_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
#endif
 protected:
  jams::Real                  temperature_;
  jams::ThermostatTemperatureProfile temperature_profile_;
  jams::MultiArray<jams::Real, 2> sigma_;
  jams::MultiArray<jams::Real, 2> noise_;

#if HAS_CUDA
  CudaStream cuda_stream_ {};
  cudaEvent_t  done_{};
  cudaEvent_t  consumed_{};
  bool has_pending_consumer_ = false;
#endif
};

#endif  // JAMS_CORE_THERMOSTAT_H
